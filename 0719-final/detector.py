from pathlib import Path
import importlib.util
import math
import sys


_LEGACY_FILE = Path(__file__).with_name("first.py")
_SPEC = importlib.util.spec_from_file_location("legacy_detector", _LEGACY_FILE)

if _SPEC is None or _SPEC.loader is None:
	raise RuntimeError(f"Cannot load detector source: {_LEGACY_FILE}")

_legacy = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _legacy
_SPEC.loader.exec_module(_legacy)

try:
	import config
except ImportError:
	config = None

if config is not None and hasattr(config, "ORANGE_HSV_RANGES"):
	_legacy.COLOR_RANGES["orange"] = config.ORANGE_HSV_RANGES
	_legacy.COLOR_TEXT["orange"] = "Orange"


def _object_stabilizer_reset(self) -> None:
	if hasattr(self, "tracks"):
		self.tracks.clear()

	if hasattr(self, "next_track_id"):
		self.next_track_id = 1


if not hasattr(_legacy.ObjectStabilizer, "reset"):
	_legacy.ObjectStabilizer.reset = _object_stabilizer_reset


FrameCandidate = _legacy.FrameCandidate
Detection = _legacy.Detection
FrameTracker = _legacy.FrameTracker
ObjectStabilizer = _legacy.ObjectStabilizer

FRAME_UPDATE_INTERVAL = _legacy.FRAME_UPDATE_INTERVAL
FRAME_WARP_WIDTH = _legacy.FRAME_WARP_WIDTH
FRAME_WARP_HEIGHT = _legacy.FRAME_WARP_HEIGHT
REGION_WARP_WIDTH = _legacy.REGION_WARP_WIDTH
REGION_WARP_HEIGHT = _legacy.REGION_WARP_HEIGHT

COLOR_RANGES = _legacy.COLOR_RANGES
COLOR_TEXT = _legacy.COLOR_TEXT
SHAPE_TEXT = _legacy.SHAPE_TEXT

average_corner_distance = _legacy.average_corner_distance
create_black_mask = _legacy.create_black_mask
analyze_middle_divider = _legacy.analyze_middle_divider
get_placement_search_roi = _legacy.get_placement_search_roi
offset_box_to_global = _legacy.offset_box_to_global
find_black_frame_candidates = _legacy.find_black_frame_candidates
warp_frame = _legacy.warp_frame
detect_objects = _legacy.detect_objects
_legacy_draw_detection = _legacy.draw_detection
draw_frame_candidates = _legacy.draw_frame_candidates
create_mask_preview = _legacy.create_mask_preview


def _is_valid_task_detection(contour, area: float, image_shape) -> bool:
	if config is None:
		return True

	image_height, image_width = image_shape[:2]
	image_area = float(image_width * image_height)

	if image_area <= 0:
		return False

	area_ratio = area / image_area

	if area_ratio < config.MIN_OBJECT_AREA_RATIO or area_ratio > config.MAX_OBJECT_AREA_RATIO:
		return False

	x, y, width, height = _legacy.cv2.boundingRect(contour)
	margin = config.OBJECT_EDGE_MARGIN_PX

	if x < margin or y < margin:
		return False

	if x + width > image_width - margin or y + height > image_height - margin:
		return False

	if width <= 0 or height <= 0:
		return False

	rect_area = float(width * height)
	fill_ratio = area / rect_area if rect_area > 0 else 0.0
	aspect_ratio = max(width, height) / max(1, min(width, height))

	if fill_ratio < config.MIN_OBJECT_FILL_RATIO:
		return False

	if aspect_ratio > config.MAX_OBJECT_ASPECT_RATIO:
		return False

	return True


def _is_basic_task_candidate(contour, area: float, image_shape) -> bool:
	if config is None:
		return True

	image_height, image_width = image_shape[:2]
	image_area = float(image_width * image_height)

	if image_area <= 0:
		return False

	if area / image_area < config.SHADOW_CORE_MIN_CONTOUR_AREA_RATIO:
		return False

	x, y, width, height = _legacy.cv2.boundingRect(contour)
	return width > 1 and height > 1 and x >= 0 and y >= 0


def _shadow_cube_fallback_enabled(enable_shadow_cube_fallback: bool, area_name: str | None) -> bool:
	if config is None:
		return False

	if not config.ENABLE_SHADOW_CUBE_FALLBACK or not enable_shadow_cube_fallback:
		return False

	return area_name in ("A", "B")


def _is_shadow_cube_edge_candidate(contour, image_shape, area_name: str | None) -> bool:
	if config is None or area_name not in ("A", "B"):
		return False

	image_width = image_shape[1]
	center_x, _ = _legacy.contour_center(contour)
	edge_ratio = float(config.SHADOW_CUBE_EDGE_ZONE_RATIO)

	if area_name == "A":
		return center_x <= image_width * edge_ratio

	return center_x >= image_width * (1.0 - edge_ratio)


def _kernel(size: int):
	size = max(1, int(size))
	return _legacy.cv2.getStructuringElement(
		_legacy.cv2.MORPH_ELLIPSE,
		(size, size),
	)


def _clip_threshold(value: float, minimum: int, maximum: int) -> int:
	return int(max(minimum, min(maximum, value)))


def _build_shadow_resistant_core_mask(hsv, base_color_mask, candidate_contour):
	if config is None:
		return None

	candidate_mask = _legacy.np.zeros(base_color_mask.shape, dtype=_legacy.np.uint8)
	_legacy.cv2.drawContours(candidate_mask, [candidate_contour], -1, 255, -1)
	sample_mask = _legacy.cv2.bitwise_and(base_color_mask, candidate_mask)
	sample_pixels = sample_mask > 0

	if int(_legacy.np.count_nonzero(sample_pixels)) < 20:
		return None

	s_values = hsv[:, :, 1][sample_pixels]
	v_values = hsv[:, :, 2][sample_pixels]
	median_s = float(_legacy.np.median(s_values))
	median_v = float(_legacy.np.median(v_values))
	s_threshold = _clip_threshold(
		median_s * config.SHADOW_CORE_S_MEDIAN_RATIO,
		config.SHADOW_CORE_S_MIN,
		config.SHADOW_CORE_S_MAX,
	)
	v_threshold = _clip_threshold(
		median_v * config.SHADOW_CORE_V_MEDIAN_RATIO,
		config.SHADOW_CORE_V_MIN,
		config.SHADOW_CORE_V_MAX,
	)

	core_mask = _legacy.cv2.inRange(
		hsv,
		_legacy.np.array((0, s_threshold, v_threshold), dtype=_legacy.np.uint8),
		_legacy.np.array((179, 255, 255), dtype=_legacy.np.uint8),
	)
	core_mask = _legacy.cv2.bitwise_and(core_mask, sample_mask)

	open_size = max(1, int(config.SHADOW_CORE_OPEN_KERNEL))
	close_size = max(1, int(config.SHADOW_CORE_CLOSE_KERNEL))

	if open_size > 1:
		core_mask = _legacy.cv2.morphologyEx(
			core_mask,
			_legacy.cv2.MORPH_OPEN,
			_kernel(open_size),
			iterations=1,
		)

	if close_size > 1:
		core_mask = _legacy.cv2.morphologyEx(
			core_mask,
			_legacy.cv2.MORPH_CLOSE,
			_kernel(close_size),
			iterations=1,
		)

	return core_mask


def _contour_fill_ratio(contour, area: float) -> float:
	rect = _legacy.cv2.minAreaRect(contour)
	(_, _), (width, height), _ = rect
	rect_area = float(width * height)

	if rect_area <= 0:
		return 0.0

	return area / rect_area


def _center_offset_ratio(original_contour, core_contour) -> float:
	original_center = _legacy.contour_center(original_contour)
	core_center = _legacy.contour_center(core_contour)
	x, y, width, height = _legacy.cv2.boundingRect(original_contour)
	scale = math.hypot(width, height)

	if scale <= 0:
		return float("inf")

	return math.dist(original_center, core_center) / scale


def _is_near_normal_cube(contour, normal_cubes) -> bool:
	center = _legacy.contour_center(contour)

	for detection in normal_cubes:
		if math.dist(center, detection.center) < 30:
			return True

		if _legacy.bounding_box_iou(contour, detection.contour) > 0.40:
			return True

	return False


def _select_shadow_core_contour(core_mask, original_contour, original_area: float, image_shape):
	image_height, image_width = image_shape[:2]
	image_area = float(image_width * image_height)
	contours, _ = _legacy.cv2.findContours(
		core_mask,
		_legacy.cv2.RETR_EXTERNAL,
		_legacy.cv2.CHAIN_APPROX_SIMPLE,
	)
	best = None
	best_score = -1.0

	for contour in contours:
		core_area = _legacy.cv2.contourArea(contour)

		if core_area <= 0 or image_area <= 0 or original_area <= 0:
			continue

		core_ratio = core_area / original_area
		area_ratio = core_area / image_area

		if core_ratio < config.SHADOW_CORE_MIN_AREA_RATIO:
			continue

		if core_ratio > config.SHADOW_CORE_MAX_AREA_RATIO:
			continue

		if area_ratio < config.SHADOW_CORE_MIN_CONTOUR_AREA_RATIO:
			continue

		(
			shape_name,
			circularity,
			aspect_ratio,
			vertex_count,
		) = _legacy.classify_shape(contour)
		fill_ratio = _contour_fill_ratio(contour, core_area)

		if aspect_ratio > config.SHADOW_CORE_MAX_ASPECT_RATIO:
			continue

		if fill_ratio < config.SHADOW_CORE_MIN_FILL_RATIO:
			continue

		if circularity > config.SHADOW_CORE_BALL_CIRCULARITY_MAX:
			continue

		offset_ratio = _center_offset_ratio(original_contour, contour)

		if offset_ratio > config.SHADOW_CORE_MAX_CENTER_OFFSET_RATIO:
			continue

		score = (
			core_area
			* max(0.2, 1.0 - abs(1.0 - aspect_ratio))
			* fill_ratio
			* (1.0 - min(offset_ratio, 0.95))
		)

		if shape_name == "cube":
			score *= 1.2
		elif shape_name == "ball":
			continue

		if score > best_score:
			best_score = score
			best = (
				contour,
				core_area,
				circularity,
				aspect_ratio,
				vertex_count,
				core_ratio,
				fill_ratio,
				area_ratio,
			)

	return best


def _make_shadow_cube_detection(color_name: str, hsv, base_mask, original_contour, original_area: float, image_shape):
	core_mask = _build_shadow_resistant_core_mask(hsv, base_mask, original_contour)

	if core_mask is None:
		return None

	core_result = _select_shadow_core_contour(
		core_mask,
		original_contour,
		original_area,
		image_shape,
	)

	if core_result is None:
		return None

	(
		core_contour,
		core_area,
		circularity,
		aspect_ratio,
		vertex_count,
		core_ratio,
		fill_ratio,
		area_ratio,
	) = core_result
	center = _legacy.contour_center(core_contour)
	detection = _legacy.Detection(
		color_name=color_name,
		shape_name="cube",
		contour=core_contour,
		center=center,
		area=core_area,
		circularity=circularity,
		aspect_ratio=aspect_ratio,
		vertex_count=vertex_count,
		score=core_area * max(circularity, 0.2) * 0.90,
	)
	detection.is_shadow_cube_fallback = True
	detection.shadow_cube_original_contour = original_contour
	detection.shadow_cube_core_ratio = core_ratio
	detection.shadow_cube_fill_ratio = fill_ratio
	detection.shadow_cube_area_ratio = area_ratio
	return detection


def _pink_lab_fallback_enabled(
	task_id: int | None,
	area_name: str | None,
	expected_shape: str | None,
	expected_color: str | None,
) -> bool:
	if config is None:
		return False

	if not config.ENABLE_PINK_LAB_CUBE_FALLBACK:
		return False

	if task_id not in config.PINK_LAB_FALLBACK_TASK_IDS:
		return False

	if area_name not in ("A", "B"):
		return False

	return expected_shape == "cube" and expected_color == "pink"


def _edge_zone_mask(image_shape, area_name: str | None, ratio: float):
	image_height, image_width = image_shape[:2]
	mask = _legacy.np.zeros((image_height, image_width), dtype=_legacy.np.uint8)
	edge_width = max(1, int(image_width * ratio))

	if area_name == "A":
		mask[:, :edge_width] = 255
	elif area_name == "B":
		mask[:, image_width - edge_width:] = 255

	return mask


def _rect_from_contour(contour) -> tuple[int, int, int, int]:
	x, y, width, height = _legacy.cv2.boundingRect(contour)
	return int(x), int(y), int(width), int(height)


def _expand_rect(rect: tuple[int, int, int, int], amount: int, image_shape) -> tuple[int, int, int, int]:
	image_height, image_width = image_shape[:2]
	x, y, width, height = rect
	left = max(0, x - amount)
	top = max(0, y - amount)
	right = min(image_width, x + width + amount)
	bottom = min(image_height, y + height + amount)
	return left, top, max(0, right - left), max(0, bottom - top)


def _rect_mask(image_shape, rect: tuple[int, int, int, int]):
	image_height, image_width = image_shape[:2]
	mask = _legacy.np.zeros((image_height, image_width), dtype=_legacy.np.uint8)
	x, y, width, height = rect

	if width <= 0 or height <= 0:
		return mask

	left = max(0, x)
	top = max(0, y)
	right = min(image_width, x + width)
	bottom = min(image_height, y + height)

	if right > left and bottom > top:
		mask[top:bottom, left:right] = 255

	return mask


def _estimate_local_lab_background(lab, valid_mask, candidate_rect, exclusion_mask):
	outer_rect = _expand_rect(
		candidate_rect,
		int(config.PINK_LAB_BACKGROUND_RING_OUTER),
		lab.shape,
	)
	inner_rect = _expand_rect(
		candidate_rect,
		int(config.PINK_LAB_BACKGROUND_RING_INNER),
		lab.shape,
	)
	outer_mask = _rect_mask(lab.shape, outer_rect)
	inner_mask = _rect_mask(lab.shape, inner_rect)
	ring_mask = _legacy.cv2.subtract(outer_mask, inner_mask)

	if valid_mask is not None:
		ring_mask = _legacy.cv2.bitwise_and(ring_mask, valid_mask)

	if exclusion_mask is not None:
		ring_mask = _legacy.cv2.bitwise_and(
			ring_mask,
			_legacy.cv2.bitwise_not(exclusion_mask),
		)

	light_mask = (lab[:, :, 0] >= config.PINK_LAB_MIN_L).astype(_legacy.np.uint8) * 255
	ring_mask = _legacy.cv2.bitwise_and(ring_mask, light_mask)
	pixels = ring_mask > 0

	if int(_legacy.np.count_nonzero(pixels)) < 40:
		return None

	background = _legacy.np.median(lab[pixels].astype(_legacy.np.float32), axis=0)
	return background, ring_mask, outer_rect, inner_rect


def _build_pink_lab_difference_mask(lab, background, search_mask):
	lab_float = lab.astype(_legacy.np.float32)
	background_l = float(background[0])
	background_a = float(background[1])
	background_b = float(background[2])
	l_delta = lab_float[:, :, 0] - background_l
	a_delta = lab_float[:, :, 1] - background_a
	b_delta = lab_float[:, :, 2] - background_b
	delta_e = _legacy.np.sqrt(l_delta * l_delta + a_delta * a_delta + b_delta * b_delta)
	mask = (
		(a_delta >= config.PINK_LAB_MIN_A_DELTA)
		& (delta_e >= config.PINK_LAB_MIN_DELTA_E)
		& (lab_float[:, :, 0] >= config.PINK_LAB_MIN_L)
		& (search_mask > 0)
	).astype(_legacy.np.uint8) * 255

	open_size = max(1, int(config.PINK_LAB_OPEN_KERNEL))
	close_size = max(1, int(config.PINK_LAB_CLOSE_KERNEL))

	if open_size > 1:
		mask = _legacy.cv2.morphologyEx(
			mask,
			_legacy.cv2.MORPH_OPEN,
			_kernel(open_size),
			iterations=1,
		)

	if close_size > 1:
		mask = _legacy.cv2.morphologyEx(
			mask,
			_legacy.cv2.MORPH_CLOSE,
			_kernel(close_size),
			iterations=1,
		)

	return mask, a_delta, delta_e


def _contour_solidity(contour, area: float) -> float:
	hull = _legacy.cv2.convexHull(contour)
	hull_area = _legacy.cv2.contourArea(hull)

	if hull_area <= 0:
		return 0.0

	return area / hull_area


def _dark_pixel_ratio(lab, contour) -> float:
	contour_mask = _legacy.np.zeros(lab.shape[:2], dtype=_legacy.np.uint8)
	_legacy.cv2.drawContours(contour_mask, [contour], -1, 255, -1)
	pixels = contour_mask > 0
	total = int(_legacy.np.count_nonzero(pixels))

	if total <= 0:
		return 1.0

	dark = int(_legacy.np.count_nonzero(lab[:, :, 0][pixels] < config.PINK_LAB_MIN_L))
	return dark / total


def _has_existing_pink_edge_cube(detections, image_shape, area_name: str | None) -> bool:
	for detection in detections:
		if detection.shape_name != "cube" or detection.color_name != "pink":
			continue

		if _is_pink_lab_edge_center(detection.center[0], image_shape, area_name):
			return True

	return False


def _is_pink_lab_edge_center(center_x: float, image_shape, area_name: str | None) -> bool:
	if config is None or area_name not in ("A", "B"):
		return False

	image_width = image_shape[1]
	edge_ratio = float(config.PINK_LAB_EDGE_ZONE_RATIO)

	if area_name == "A":
		return center_x <= image_width * edge_ratio

	return center_x >= image_width * (1.0 - edge_ratio)


def _candidate_rects_from_pink_mask(pink_mask, search_mask, image_shape):
	rects = []
	mask = _legacy.cv2.bitwise_and(pink_mask, search_mask)
	contours, _ = _legacy.cv2.findContours(
		mask,
		_legacy.cv2.RETR_EXTERNAL,
		_legacy.cv2.CHAIN_APPROX_SIMPLE,
	)

	for contour in contours:
		area = _legacy.cv2.contourArea(contour)

		if area < max(40.0, _legacy.MIN_OBJECT_AREA * 0.08):
			continue

		rects.append(_expand_rect(_rect_from_contour(contour), 14, image_shape))

	return rects


def _candidate_rects_from_lab_difference(lab, valid_mask, search_mask, pink_mask):
	background_mask = _legacy.cv2.bitwise_and(search_mask, valid_mask) if valid_mask is not None else search_mask.copy()
	background_mask = _legacy.cv2.bitwise_and(background_mask, _legacy.cv2.bitwise_not(pink_mask))
	background_mask = _legacy.cv2.bitwise_and(
		background_mask,
		((lab[:, :, 0] >= config.PINK_LAB_MIN_L).astype(_legacy.np.uint8) * 255),
	)
	pixels = background_mask > 0

	if int(_legacy.np.count_nonzero(pixels)) < 80:
		return []

	background = _legacy.np.median(lab[pixels].astype(_legacy.np.float32), axis=0)
	rough_mask, _, _ = _build_pink_lab_difference_mask(lab, background, search_mask)

	if valid_mask is not None:
		rough_mask = _legacy.cv2.bitwise_and(rough_mask, valid_mask)

	contours, _ = _legacy.cv2.findContours(
		rough_mask,
		_legacy.cv2.RETR_EXTERNAL,
		_legacy.cv2.CHAIN_APPROX_SIMPLE,
	)
	rects = []

	for contour in contours:
		area = _legacy.cv2.contourArea(contour)

		if area < max(50.0, lab.shape[0] * lab.shape[1] * config.PINK_LAB_MIN_AREA_RATIO * 0.50):
			continue

		rects.append(_expand_rect(_rect_from_contour(contour), 10, lab.shape))

	return rects


def _dedupe_rects(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
	result = []

	for rect in rects:
		x, y, width, height = rect
		if width <= 1 or height <= 1:
			continue

		is_duplicate = False

		for accepted in result:
			ax, ay, aw, ah = accepted
			left = max(x, ax)
			top = max(y, ay)
			right = min(x + width, ax + aw)
			bottom = min(y + height, ay + ah)
			intersection = max(0, right - left) * max(0, bottom - top)
			union = width * height + aw * ah - intersection

			if union > 0 and intersection / union > 0.55:
				is_duplicate = True
				break

		if not is_duplicate:
			result.append(rect)

	return result


def _select_pink_lab_core_contour(lab, lab_mask, candidate_rect, background, a_delta, delta_e):
	image_area = float(lab.shape[0] * lab.shape[1])
	candidate_area = float(candidate_rect[2] * candidate_rect[3])
	rect_mask = _rect_mask(lab.shape, candidate_rect)
	mask = _legacy.cv2.bitwise_and(lab_mask, rect_mask)
	contours, _ = _legacy.cv2.findContours(
		mask,
		_legacy.cv2.RETR_EXTERNAL,
		_legacy.cv2.CHAIN_APPROX_SIMPLE,
	)
	best = None
	best_score = -1.0

	for contour in contours:
		area = _legacy.cv2.contourArea(contour)

		if area <= 0 or image_area <= 0 or candidate_area <= 0:
			continue

		if area / image_area < config.PINK_LAB_MIN_AREA_RATIO:
			continue

		if area / candidate_area < config.PINK_LAB_MIN_CORE_TO_CANDIDATE_RATIO:
			continue

		(
			shape_name,
			circularity,
			aspect_ratio,
			vertex_count,
		) = _legacy.classify_shape(contour)
		fill_ratio = _contour_fill_ratio(contour, area)
		solidity = _contour_solidity(contour, area)

		if aspect_ratio > config.PINK_LAB_MAX_ASPECT_RATIO:
			continue

		if fill_ratio < config.PINK_LAB_MIN_FILL_RATIO:
			continue

		if solidity < config.PINK_LAB_MIN_SOLIDITY:
			continue

		if circularity > config.PINK_LAB_BALL_CIRCULARITY_MAX or shape_name == "ball":
			continue

		if _dark_pixel_ratio(lab, contour) > config.PINK_LAB_MAX_DARK_PIXEL_RATIO:
			continue

		offset_ratio = _center_offset_ratio(
			_legacy.np.array(
				[
					[[candidate_rect[0], candidate_rect[1]]],
					[[candidate_rect[0] + candidate_rect[2], candidate_rect[1]]],
					[[candidate_rect[0] + candidate_rect[2], candidate_rect[1] + candidate_rect[3]]],
					[[candidate_rect[0], candidate_rect[1] + candidate_rect[3]]],
				],
				dtype=_legacy.np.int32,
			),
			contour,
		)

		if offset_ratio > config.PINK_LAB_MAX_CENTER_OFFSET_RATIO:
			continue

		contour_mask = _legacy.np.zeros(lab.shape[:2], dtype=_legacy.np.uint8)
		_legacy.cv2.drawContours(contour_mask, [contour], -1, 255, -1)
		pixels = contour_mask > 0
		median_a_delta = float(_legacy.np.median(a_delta[pixels]))
		median_delta_e = float(_legacy.np.median(delta_e[pixels]))
		score = area * fill_ratio * solidity * (1.0 - min(offset_ratio, 0.95))

		if shape_name == "cube":
			score *= 1.15

		if score > best_score:
			best_score = score
			best = (
				contour,
				area,
				circularity,
				aspect_ratio,
				vertex_count,
				fill_ratio,
				solidity,
				median_a_delta,
				median_delta_e,
			)

	return best


def _detect_pink_cube_with_lab_fallback(
	image_bgr,
	valid_mask,
	area_name: str | None,
	pink_mask,
	existing_detections,
	task_id: int | None,
	expected_shape: str | None,
	expected_color: str | None,
):
	if not _pink_lab_fallback_enabled(task_id, area_name, expected_shape, expected_color):
		return None

	if _has_existing_pink_edge_cube(existing_detections, image_bgr.shape, area_name):
		return None

	lab = _legacy.cv2.cvtColor(image_bgr, _legacy.cv2.COLOR_BGR2LAB)
	search_mask = _edge_zone_mask(
		image_bgr.shape,
		area_name,
		float(config.PINK_LAB_EDGE_ZONE_RATIO),
	)

	if valid_mask is not None:
		search_mask = _legacy.cv2.bitwise_and(search_mask, valid_mask)

	pink_mask = pink_mask if pink_mask is not None else _legacy.np.zeros(search_mask.shape, dtype=_legacy.np.uint8)
	candidate_rects = _candidate_rects_from_pink_mask(pink_mask, search_mask, image_bgr.shape)
	candidate_rects.extend(
		_candidate_rects_from_lab_difference(
			lab,
			valid_mask,
			search_mask,
			pink_mask,
		)
	)
	candidate_rects = _dedupe_rects(candidate_rects)
	best_detection = None
	best_score = -1.0

	for rect in candidate_rects:
		rect_center_x = rect[0] + rect[2] / 2.0

		if not _is_pink_lab_edge_center(rect_center_x, image_bgr.shape, area_name):
			continue

		background_result = _estimate_local_lab_background(
			lab,
			valid_mask,
			rect,
			pink_mask,
		)

		if background_result is None:
			continue

		background, ring_mask, outer_rect, inner_rect = background_result
		local_search_mask = _legacy.cv2.bitwise_and(search_mask, _rect_mask(image_bgr.shape, outer_rect))
		lab_mask, a_delta, delta_e = _build_pink_lab_difference_mask(
			lab,
			background,
			local_search_mask,
		)

		if valid_mask is not None:
			lab_mask = _legacy.cv2.bitwise_and(lab_mask, valid_mask)

		core_result = _select_pink_lab_core_contour(
			lab,
			lab_mask,
			rect,
			background,
			a_delta,
			delta_e,
		)

		if core_result is None:
			continue

		(
			contour,
			area,
			circularity,
			aspect_ratio,
			vertex_count,
			fill_ratio,
			solidity,
			median_a_delta,
			median_delta_e,
		) = core_result
		center = _legacy.contour_center(contour)
		detection = _legacy.Detection(
			color_name="pink",
			shape_name="cube",
			contour=contour,
			center=center,
			area=area,
			circularity=circularity,
			aspect_ratio=aspect_ratio,
			vertex_count=vertex_count,
			score=area * max(circularity, 0.2) * fill_ratio * solidity * 0.85,
		)
		detection.is_pink_lab_fallback = True
		detection.pink_lab_background = tuple(float(item) for item in background)
		detection.pink_lab_a_delta = median_a_delta
		detection.pink_lab_delta_e = median_delta_e
		detection.pink_lab_fill_ratio = fill_ratio
		detection.pink_lab_solidity = solidity
		detection.pink_lab_outer_rect = outer_rect
		detection.pink_lab_inner_rect = inner_rect
		detection.pink_lab_mask = lab_mask
		detection.pink_lab_ring_mask = ring_mask

		if detection.score > best_score:
			best_score = detection.score
			best_detection = detection

	return best_detection


def _apply_valid_mask(mask, valid_mask):
	if valid_mask is None:
		return mask

	return _legacy.cv2.bitwise_and(
		mask,
		valid_mask,
	)


def _clean_task_color_mask(mask):
	if config is None:
		return mask

	open_size = max(1, int(getattr(config, "COLOR_MASK_OPEN_KERNEL", 3)))
	close_size = max(1, int(getattr(config, "COLOR_MASK_CLOSE_KERNEL", 5)))

	if open_size > 1:
		open_kernel = _legacy.cv2.getStructuringElement(
			_legacy.cv2.MORPH_ELLIPSE,
			(open_size, open_size),
		)
		mask = _legacy.cv2.morphologyEx(
			mask,
			_legacy.cv2.MORPH_OPEN,
			open_kernel,
			iterations=1,
		)

	if close_size > 1:
		close_kernel = _legacy.cv2.getStructuringElement(
			_legacy.cv2.MORPH_ELLIPSE,
			(close_size, close_size),
		)
		mask = _legacy.cv2.morphologyEx(
			mask,
			_legacy.cv2.MORPH_CLOSE,
			close_kernel,
			iterations=1,
		)

	return mask


def detect_all_objects(
	image,
	valid_mask=None,
	area_name: str | None = None,
	enable_shadow_cube_fallback: bool = False,
	task_id: int | None = None,
	expected_shape: str | None = None,
	expected_color: str | None = None,
):
	hsv = _legacy.cv2.cvtColor(image, _legacy.cv2.COLOR_BGR2HSV)
	detections = []
	masks = {}
	shadow_candidates = []
	pink_mask = None

	for color_name, ranges in _legacy.COLOR_RANGES.items():
		mask = _legacy.build_color_mask(hsv, ranges)
		mask = _apply_valid_mask(mask, valid_mask)
		mask = _clean_task_color_mask(mask)
		masks[color_name] = mask

		if color_name == "pink":
			pink_mask = mask

		contours, _ = _legacy.cv2.findContours(
			mask,
			_legacy.cv2.RETR_EXTERNAL,
			_legacy.cv2.CHAIN_APPROX_SIMPLE,
		)

		for contour in contours:
			area = _legacy.cv2.contourArea(contour)

			if area < _legacy.MIN_OBJECT_AREA:
				continue

			if not _is_basic_task_candidate(contour, area, image.shape):
				continue

			is_valid_normal = _is_valid_task_detection(contour, area, image.shape)

			if is_valid_normal:
				(
					shape_name,
					circularity,
					aspect_ratio,
					vertex_count,
				) = _legacy.classify_shape(contour)
				center = _legacy.contour_center(contour)
				score = area * max(circularity, 0.2)

				detections.append(
					_legacy.Detection(
						color_name=color_name,
						shape_name=shape_name,
						contour=contour,
						center=center,
						area=area,
						circularity=circularity,
						aspect_ratio=aspect_ratio,
						vertex_count=vertex_count,
						score=score,
					)
				)

			if (
				_shadow_cube_fallback_enabled(enable_shadow_cube_fallback, area_name)
				and _is_shadow_cube_edge_candidate(contour, image.shape, area_name)
			):
				shadow_candidates.append((color_name, mask, contour, area))

	normal_cubes = [
		detection
		for detection in detections
		if detection.shape_name == "cube"
	]

	for color_name, mask, contour, area in shadow_candidates:
		if _is_near_normal_cube(contour, normal_cubes):
			continue

		fallback_detection = _make_shadow_cube_detection(
			color_name,
			hsv,
			mask,
			contour,
			area,
			image.shape,
		)

		if fallback_detection is None:
			continue

		detections.append(fallback_detection)
		normal_cubes.append(fallback_detection)

	pink_lab_detection = _detect_pink_cube_with_lab_fallback(
		image,
		valid_mask,
		area_name,
		pink_mask,
		detections,
		task_id,
		expected_shape,
		expected_color,
	)

	if pink_lab_detection is not None and not _is_near_normal_cube(pink_lab_detection.contour, normal_cubes):
		detections.append(pink_lab_detection)
		normal_cubes.append(pink_lab_detection)

	detections.sort(
		key=lambda item: item.score,
		reverse=True,
	)
	detections = _legacy.remove_duplicate_detections(detections)

	return detections[:6], masks


def draw_detection(image, detection, index: int) -> None:
	_legacy_draw_detection(image, detection, index)

	if config is None:
		return

	if (
		config.DEBUG_DRAW_ENABLED
		and config.DEBUG_DRAW_SHADOW_CUBE_FALLBACK
		and getattr(detection, "is_shadow_cube_fallback", False)
	):
		original_contour = getattr(detection, "shadow_cube_original_contour", None)

		if original_contour is not None:
			_legacy.cv2.drawContours(
				image,
				[original_contour],
				-1,
				(0, 255, 255),
				1,
			)

		rect = _legacy.cv2.minAreaRect(detection.contour)
		box = _legacy.cv2.boxPoints(rect).astype(_legacy.np.int32)
		center_x, center_y = detection.center
		_legacy.cv2.polylines(image, [box], True, (255, 0, 255), 2)
		_legacy.cv2.circle(image, (center_x, center_y), 6, (255, 0, 255), -1)
		_legacy.cv2.putText(
			image,
			"shadow-cube",
			(max(center_x - 70, 5), center_y + 38),
			_legacy.cv2.FONT_HERSHEY_SIMPLEX,
			0.50,
			(255, 0, 255),
			1,
			_legacy.cv2.LINE_AA,
		)
		_legacy.cv2.putText(
			image,
			(
				f"core={getattr(detection, 'shadow_cube_core_ratio', 0.0):.2f} "
				f"fill={getattr(detection, 'shadow_cube_fill_ratio', 0.0):.2f} "
				f"area={getattr(detection, 'shadow_cube_area_ratio', 0.0):.3f}"
			),
			(max(center_x - 88, 5), center_y + 58),
			_legacy.cv2.FONT_HERSHEY_SIMPLEX,
			0.42,
			(255, 0, 255),
			1,
			_legacy.cv2.LINE_AA,
		)

	if (
		config.DEBUG_DRAW_ENABLED
		and config.DEBUG_DRAW_PINK_LAB_FALLBACK
		and getattr(detection, "is_pink_lab_fallback", False)
	):
		center_x, center_y = detection.center
		outer_rect = getattr(detection, "pink_lab_outer_rect", None)
		inner_rect = getattr(detection, "pink_lab_inner_rect", None)

		for rect, color in ((outer_rect, (255, 255, 0)), (inner_rect, (0, 255, 255))):
			if rect is None:
				continue

			x, y, width, height = rect
			_legacy.cv2.rectangle(
				image,
				(int(x), int(y)),
				(int(x + width), int(y + height)),
				color,
				1,
			)

		rect = _legacy.cv2.minAreaRect(detection.contour)
		box = _legacy.cv2.boxPoints(rect).astype(_legacy.np.int32)
		_legacy.cv2.polylines(image, [box], True, (255, 0, 180), 2)
		_legacy.cv2.circle(image, (center_x, center_y), 6, (255, 0, 180), -1)
		_legacy.cv2.putText(
			image,
			"pink-lab",
			(max(center_x - 55, 5), center_y + 38),
			_legacy.cv2.FONT_HERSHEY_SIMPLEX,
			0.50,
			(255, 0, 180),
			1,
			_legacy.cv2.LINE_AA,
		)
		_legacy.cv2.putText(
			image,
			(
				f"a={getattr(detection, 'pink_lab_a_delta', 0.0):.1f} "
				f"de={getattr(detection, 'pink_lab_delta_e', 0.0):.1f} "
				f"r={detection.aspect_ratio:.2f} "
				f"f={getattr(detection, 'pink_lab_fill_ratio', 0.0):.2f} "
				f"s={getattr(detection, 'pink_lab_solidity', 0.0):.2f}"
			),
			(max(center_x - 110, 5), center_y + 58),
			_legacy.cv2.FONT_HERSHEY_SIMPLEX,
			0.38,
			(255, 0, 180),
			1,
			_legacy.cv2.LINE_AA,
		)


def reset_object_stabilizer(stabilizer) -> None:
	if hasattr(stabilizer, "reset"):
		stabilizer.reset()


def get_show_frame_candidates() -> bool:
	if config is not None:
		return bool(config.DEBUG_DRAW_ENABLED and config.DEBUG_DRAW_ALL_CANDIDATES)

	return bool(_legacy.SHOW_FRAME_CANDIDATES)


def set_show_frame_candidates(value: bool) -> None:
	if config is not None:
		config.DEBUG_DRAW_ENABLED = value
		config.DEBUG_DRAW_ALL_CANDIDATES = value
		config.DEBUG_DRAW_CANDIDATE_TEXT = value
		config.DEBUG_DRAW_ALL_PAIRS = value
		config.DEBUG_DRAW_PAIR_TEXT = value
		config.DEBUG_DRAW_ROI = value
		config.SHOW_SOURCE_CANDIDATES_WINDOW = value

	_legacy.SHOW_FRAME_CANDIDATES = value


def toggle_show_frame_candidates() -> bool:
	next_value = not get_show_frame_candidates()
	set_show_frame_candidates(next_value)
	return next_value

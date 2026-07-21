from __future__ import annotations

from dataclasses import dataclass

import config
import serial_protocol as protocol


@dataclass
class SceneDetection:
	shape: str
	color: str
	center_px: tuple[int, int]
	global_center_x: float
	global_center_y: float
	area: str
	score: float
	stable_frames: int = 1
	standard_center_x: float | None = None
	standard_center_y: float | None = None
	object_axis_angle_deg: float | None = None
	is_shadow_cube_fallback: bool = False
	is_pink_lab_fallback: bool = False


@dataclass
class PlacementRegion:
	box: object
	score: float
	center: tuple[float, float]
	area: str
	divider_x: int | None = None
	divider_source: str = "center_fallback"
	divider_score: float = 0.0
	vertical_coverage: float = 0.0


@dataclass
class AreaView:
	name: str
	region: PlacementRegion
	source_crop: object
	valid_mask: object
	result_image: object
	crop_offset: tuple[int, int]
	crop_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
	homography: object | None = None
	inverse_homography: object | None = None
	warp_image: object | None = None


@dataclass
class PairEvaluation:
	left_index: int
	right_index: int
	left_candidate: object
	right_candidate: object
	pair_score: float
	area_similarity: float
	width_similarity: float
	height_similarity: float
	center_y_similarity: float
	gap_ratio: float
	divider_score: float
	accepted: bool
	reject_reason: str = ""


@dataclass
class PlacementRegions:
	region_a: PlacementRegion
	region_b: PlacementRegion
	area_a_image: object
	area_b_image: object
	pair_score: float
	pair_evaluation: PairEvaluation
	selected_left_candidate: object
	selected_right_candidate: object
	area_a_view: AreaView | None = None
	area_b_view: AreaView | None = None
	area_a_warp: object | None = None
	area_b_warp: object | None = None


@dataclass
class SceneBuildResult:
	detections: list[SceneDetection]
	placement: PlacementRegions | None
	area_a_result: object | None = None
	area_b_result: object | None = None


def _order_points(points: np.ndarray) -> np.ndarray:
	import numpy as np

	points = np.asarray(points, dtype=np.float32)
	ordered = np.zeros((4, 2), dtype=np.float32)
	point_sum = points.sum(axis=1)
	point_diff = np.diff(points, axis=1).reshape(-1)
	ordered[0] = points[np.argmin(point_sum)]
	ordered[2] = points[np.argmax(point_sum)]
	ordered[1] = points[np.argmin(point_diff)]
	ordered[3] = points[np.argmax(point_diff)]
	return ordered


def _point_tuple(point) -> tuple[float, float]:
	return float(point[0]), float(point[1])


def _order_points_plain(points) -> list[tuple[float, float]]:
	plain_points = [_point_tuple(point) for point in points]
	point_sum = [point[0] + point[1] for point in plain_points]
	point_diff = [point[0] - point[1] for point in plain_points]
	return [
		plain_points[point_sum.index(min(point_sum))],
		plain_points[point_diff.index(max(point_diff))],
		plain_points[point_sum.index(max(point_sum))],
		plain_points[point_diff.index(min(point_diff))],
	]


def polygon_center(points) -> tuple[float, float]:
	plain_points = [_point_tuple(point) for point in points]
	return (
		sum(point[0] for point in plain_points) / len(plain_points),
		sum(point[1] for point in plain_points) / len(plain_points),
	)


def _candidate_bounds(candidate) -> tuple[float, float, float, float]:
	points = [_point_tuple(point) for point in candidate.box]
	x_values = [point[0] for point in points]
	y_values = [point[1] for point in points]
	return min(x_values), min(y_values), max(x_values), max(y_values)


def _candidate_area(candidate) -> float:
	left, top, right, bottom = _candidate_bounds(candidate)
	return max(0.0, right - left) * max(0.0, bottom - top)


def _bbox_iou(left_candidate, right_candidate) -> float:
	left_a, top_a, right_a, bottom_a = _candidate_bounds(left_candidate)
	left_b, top_b, right_b, bottom_b = _candidate_bounds(right_candidate)
	inter_left = max(left_a, left_b)
	inter_top = max(top_a, top_b)
	inter_right = min(right_a, right_b)
	inter_bottom = min(bottom_a, bottom_b)

	if inter_right <= inter_left or inter_bottom <= inter_top:
		return 0.0

	intersection = (inter_right - inter_left) * (inter_bottom - inter_top)
	area_a = _candidate_area(left_candidate)
	area_b = _candidate_area(right_candidate)
	union = area_a + area_b - intersection
	return intersection / union if union > 0 else 0.0


def _center_distance(left_candidate, right_candidate) -> float:
	import math

	left_x, left_y = _point_tuple(left_candidate.center)
	right_x, right_y = _point_tuple(right_candidate.center)
	return math.hypot(left_x - right_x, left_y - right_y)


def _size_ratio_diff(left_candidate, right_candidate) -> float:
	area_a = _candidate_area(left_candidate)
	area_b = _candidate_area(right_candidate)

	if area_a <= 0 or area_b <= 0:
		return 1.0

	return abs(area_a - area_b) / max(area_a, area_b)


def _horizontal_gap(left_candidate, right_candidate) -> float:
	left_x = float(left_candidate.center[0])
	right_x = float(right_candidate.center[0])
	return abs(left_x - right_x)


def _candidate_width_height(candidate) -> tuple[float, float]:
	left, top, right, bottom = _candidate_bounds(candidate)
	return max(0.0, right - left), max(0.0, bottom - top)


def _ratio_score(candidate) -> float:
	ratio = float(getattr(candidate, "ratio", 0.0))
	return max(
		0.0,
		1.0 - abs(ratio - config.INNER_REGION_EXPECTED_RATIO) / 0.35,
	)


def _relative_diff(left_value: float, right_value: float) -> float:
	if left_value <= 0 or right_value <= 0:
		return 1.0

	return abs(left_value - right_value) / max(left_value, right_value)


def _is_duplicate_region(candidate, selected_candidate) -> bool:
	iou = _bbox_iou(candidate, selected_candidate)

	if iou >= config.BLACK_FRAME_NMS_IOU_THRESHOLD:
		return True

	if (
		_center_distance(candidate, selected_candidate) < config.BLACK_FRAME_MIN_CENTER_DISTANCE_PX
		and _size_ratio_diff(candidate, selected_candidate) <= config.BLACK_FRAME_MAX_SIZE_RATIO_DIFF
	):
		return True

	return False


def _interpolate_points(left, right, ratio: float):
	left_x, left_y = _point_tuple(left)
	right_x, right_y = _point_tuple(right)
	return (
		left_x + (right_x - left_x) * ratio,
		left_y + (right_y - left_y) * ratio,
	)


def _make_regions_from_pair(left_candidate, right_candidate, pair_score: float) -> list[PlacementRegion]:
	return [
		PlacementRegion(
			box=left_candidate.box,
			score=pair_score,
			center=polygon_center(left_candidate.box),
			area="A",
			divider_x=int((left_candidate.center[0] + right_candidate.center[0]) / 2.0),
			divider_source="pair_gap",
			divider_score=pair_score,
			vertical_coverage=0.0,
		),
		PlacementRegion(
			box=right_candidate.box,
			score=pair_score,
			center=polygon_center(right_candidate.box),
			area="B",
			divider_x=int((left_candidate.center[0] + right_candidate.center[0]) / 2.0),
			divider_source="pair_gap",
			divider_score=pair_score,
			vertical_coverage=0.0,
		),
	]


def _make_rejected_pair(
	left_index: int,
	right_index: int,
	left_candidate,
	right_candidate,
	reason: str,
) -> PairEvaluation:
	return PairEvaluation(
		left_index=left_index,
		right_index=right_index,
		left_candidate=left_candidate,
		right_candidate=right_candidate,
		pair_score=0.0,
		area_similarity=0.0,
		width_similarity=0.0,
		height_similarity=0.0,
		center_y_similarity=0.0,
		gap_ratio=0.0,
		divider_score=0.0,
		accepted=False,
		reject_reason=reason,
	)


def _score_candidate_pair(
	left_candidate,
	right_candidate,
	left_index: int = 0,
	right_index: int = 0,
) -> PairEvaluation:
	left_x = float(left_candidate.center[0])
	right_x = float(right_candidate.center[0])

	if left_x >= right_x:
		return _make_rejected_pair(
			left_index,
			right_index,
			left_candidate,
			right_candidate,
			"not_left_right",
		)

	left_ratio = float(getattr(left_candidate, "ratio", 0.0))
	right_ratio = float(getattr(right_candidate, "ratio", 0.0))

	if not config.INNER_REGION_RATIO_MIN <= left_ratio <= config.INNER_REGION_RATIO_MAX:
		return _make_rejected_pair(
			left_index,
			right_index,
			left_candidate,
			right_candidate,
			"left_ratio",
		)

	if not config.INNER_REGION_RATIO_MIN <= right_ratio <= config.INNER_REGION_RATIO_MAX:
		return _make_rejected_pair(
			left_index,
			right_index,
			left_candidate,
			right_candidate,
			"right_ratio",
		)

	left_width, left_height = _candidate_width_height(left_candidate)
	right_width, right_height = _candidate_width_height(right_candidate)
	left_area = _candidate_area(left_candidate)
	right_area = _candidate_area(right_candidate)

	if left_width <= 0 or left_height <= 0 or right_width <= 0 or right_height <= 0:
		return _make_rejected_pair(
			left_index,
			right_index,
			left_candidate,
			right_candidate,
			"bad_size",
		)

	area_diff = _relative_diff(left_area, right_area)
	width_diff = _relative_diff(left_width, right_width)
	height_diff = _relative_diff(left_height, right_height)
	center_y_diff = abs(float(left_candidate.center[1]) - float(right_candidate.center[1]))
	avg_height = (left_height + right_height) / 2.0
	center_y_ratio = center_y_diff / max(1.0, avg_height)

	left_bounds = _candidate_bounds(left_candidate)
	right_bounds = _candidate_bounds(right_candidate)
	gap = right_bounds[0] - left_bounds[2]
	avg_width = (left_width + right_width) / 2.0
	gap_ratio = gap / max(1.0, avg_width)
	area_similarity = max(0.0, 1.0 - area_diff)
	width_similarity = max(0.0, 1.0 - width_diff)
	height_similarity = max(0.0, 1.0 - height_diff)
	center_y_similarity = max(0.0, 1.0 - center_y_ratio)
	divider_score = max(0.0, 1.0 - abs(gap_ratio))

	if area_diff > config.PAIR_MAX_AREA_DIFF_RATIO:
		return PairEvaluation(
			left_index,
			right_index,
			left_candidate,
			right_candidate,
			0.0,
			area_similarity,
			width_similarity,
			height_similarity,
			center_y_similarity,
			gap_ratio,
			divider_score,
			False,
			"area_diff",
		)

	if width_diff > config.PAIR_MAX_WIDTH_DIFF_RATIO:
		return PairEvaluation(
			left_index,
			right_index,
			left_candidate,
			right_candidate,
			0.0,
			area_similarity,
			width_similarity,
			height_similarity,
			center_y_similarity,
			gap_ratio,
			divider_score,
			False,
			"width_diff",
		)

	if height_diff > config.PAIR_MAX_HEIGHT_DIFF_RATIO:
		return PairEvaluation(
			left_index,
			right_index,
			left_candidate,
			right_candidate,
			0.0,
			area_similarity,
			width_similarity,
			height_similarity,
			center_y_similarity,
			gap_ratio,
			divider_score,
			False,
			"height_diff",
		)

	if center_y_ratio > config.PAIR_MAX_CENTER_Y_DIFF_RATIO:
		return PairEvaluation(
			left_index,
			right_index,
			left_candidate,
			right_candidate,
			0.0,
			area_similarity,
			width_similarity,
			height_similarity,
			center_y_similarity,
			gap_ratio,
			divider_score,
			False,
			"center_y",
		)

	if gap_ratio < config.PAIR_MIN_HORIZONTAL_GAP_RATIO:
		return PairEvaluation(
			left_index,
			right_index,
			left_candidate,
			right_candidate,
			0.0,
			area_similarity,
			width_similarity,
			height_similarity,
			center_y_similarity,
			gap_ratio,
			divider_score,
			False,
			"gap_too_small",
		)

	if gap_ratio > config.PAIR_MAX_HORIZONTAL_GAP_RATIO:
		return PairEvaluation(
			left_index,
			right_index,
			left_candidate,
			right_candidate,
			0.0,
			area_similarity,
			width_similarity,
			height_similarity,
			center_y_similarity,
			gap_ratio,
			divider_score,
			False,
			"gap_too_large",
		)

	score = (
		0.05 * ((_ratio_score(left_candidate) + _ratio_score(right_candidate)) / 2.0)
		+ 0.24 * area_similarity
		+ 0.22 * width_similarity
		+ 0.20 * height_similarity
		+ 0.19 * center_y_similarity
		+ 0.10 * divider_score
	)

	return PairEvaluation(
		left_index=left_index,
		right_index=right_index,
		left_candidate=left_candidate,
		right_candidate=right_candidate,
		pair_score=score,
		area_similarity=area_similarity,
		width_similarity=width_similarity,
		height_similarity=height_similarity,
		center_y_similarity=center_y_similarity,
		gap_ratio=gap_ratio,
		divider_score=divider_score,
		accepted=True,
	)


def evaluate_candidate_pairs(candidates) -> list[PairEvaluation]:
	valid_candidates = [
		(original_index + 1, candidate)
		for original_index, candidate in enumerate(candidates)
		if getattr(candidate, "box", None) is not None
		and float(getattr(candidate, "score", 0.0)) >= config.PAIR_MIN_SCORE
	]
	evaluations = []

	for left_offset, left_item in enumerate(valid_candidates):
		for right_item in valid_candidates[left_offset + 1:]:
			candidate_pair = sorted(
				[left_item, right_item],
				key=lambda item: item[1].center[0],
			)
			evaluations.append(
				_score_candidate_pair(
					candidate_pair[0][1],
					candidate_pair[1][1],
					candidate_pair[0][0],
					candidate_pair[1][0],
				)
			)

	evaluations.sort(
		key=lambda item: item.pair_score,
		reverse=True,
	)
	return evaluations


def select_inner_region_pair(candidates):
	best_pair = None

	for evaluation in evaluate_candidate_pairs(candidates):
		if evaluation.accepted:
			best_pair = evaluation
			break

	return best_pair


def select_two_regions(candidates) -> list[PlacementRegion]:
	evaluation = select_inner_region_pair(candidates)

	if evaluation is None:
		return []

	return _make_regions_from_pair(
		evaluation.left_candidate,
		evaluation.right_candidate,
		evaluation.pair_score,
	)


def detect_middle_divider_x(warped_frame) -> tuple[int, str, float]:
	import detector

	mask = detector.create_black_mask(warped_frame)
	analysis = detector.analyze_middle_divider(mask)
	_, width = mask.shape[:2]

	if not analysis.found:
		return width // 2, "center_fallback", analysis.score

	return analysis.x, "detected", analysis.score


def select_placement_regions(frame, candidates) -> list[PlacementRegion]:
	return select_two_regions(candidates)


def crop_selected_region(frame, box):
	import cv2
	import numpy as np

	x, y, width, height = cv2.boundingRect(
		np.asarray(box, dtype=np.int32),
	)
	frame_height, frame_width = frame.shape[:2]
	x = max(0, x)
	y = max(0, y)
	width = min(width, frame_width - x)
	height = min(height, frame_height - y)

	if width <= 0 or height <= 0:
		empty = frame[0:0, 0:0].copy()
		return empty, (x, y, 0, 0)

	crop = frame[y:y + height, x:x + width].copy()
	return crop, (x, y, width, height)


def crop_polygon_region(frame, box):
	import cv2
	import numpy as np

	points = np.asarray(box, dtype=np.float32)
	x, y, width, height = cv2.boundingRect(points.astype(np.int32))
	frame_height, frame_width = frame.shape[:2]
	x = max(0, x)
	y = max(0, y)
	width = min(width, frame_width - x)
	height = min(height, frame_height - y)

	if width <= 0 or height <= 0:
		empty = frame[0:0, 0:0].copy()
		return empty, np.zeros((0, 0), dtype=np.uint8), (x, y)

	source_crop = frame[y:y + height, x:x + width].copy()
	local_points = points.copy()
	local_points[:, 0] -= x
	local_points[:, 1] -= y
	mask = np.zeros((height, width), dtype=np.uint8)
	cv2.fillConvexPoly(
		mask,
		local_points.astype(np.int32),
		255,
	)

	erode_px = max(0, int(getattr(config, "VALID_REGION_ERODE_PX", 0)))
	if erode_px > 0 and mask.size > 0:
		kernel = cv2.getStructuringElement(
			cv2.MORPH_RECT,
			(erode_px * 2 + 1, erode_px * 2 + 1),
		)
		mask = cv2.erode(
			mask,
			kernel,
			iterations=1,
		)

	source_crop = cv2.bitwise_and(
		source_crop,
		source_crop,
		mask=mask,
	)
	return source_crop, mask, (x, y)


def _build_area_homography(region: PlacementRegion):
	import cv2
	import detector
	import numpy as np

	src = _order_points(region.box)
	dst = np.array(
		[
			[0, 0],
			[detector.REGION_WARP_WIDTH - 1, 0],
			[detector.REGION_WARP_WIDTH - 1, detector.REGION_WARP_HEIGHT - 1],
			[0, detector.REGION_WARP_HEIGHT - 1],
		],
		dtype=np.float32,
	)
	homography = cv2.getPerspectiveTransform(src, dst)
	inverse_homography = cv2.getPerspectiveTransform(dst, src)
	return homography, inverse_homography


def _build_area_view(name: str, frame, region: PlacementRegion) -> AreaView:
	import detector
	import numpy as np

	source_crop, crop_rect = crop_selected_region(
		frame,
		region.box,
	)
	crop_offset = (
		crop_rect[0],
		crop_rect[1],
	)
	valid_mask = np.full(
		source_crop.shape[:2],
		255,
		dtype=np.uint8,
	)
	result_image = source_crop.copy()
	homography, inverse_homography = _build_area_homography(region)
	warp_image = detector.warp_frame(
		frame,
		region.box,
		detector.REGION_WARP_WIDTH,
		detector.REGION_WARP_HEIGHT,
	)
	return AreaView(
		name=name,
		region=region,
		source_crop=source_crop,
		valid_mask=valid_mask,
		result_image=result_image,
		crop_offset=crop_offset,
		crop_rect=crop_rect,
		homography=homography,
		inverse_homography=inverse_homography,
		warp_image=warp_image,
	)


def build_placement_regions(frame, candidates) -> PlacementRegions | None:
	evaluation = select_inner_region_pair(candidates)

	if evaluation is None:
		return None

	regions = _make_regions_from_pair(
		evaluation.left_candidate,
		evaluation.right_candidate,
		evaluation.pair_score,
	)
	area_a_view = _build_area_view(
		"A",
		frame,
		regions[0],
	)
	area_b_view = _build_area_view(
		"B",
		frame,
		regions[1],
	)

	if config.VERBOSE_VISION_LOG:
		try:
			import numpy as np
			shares_memory = np.shares_memory(
				area_a_view.source_crop,
				area_b_view.source_crop,
			)
		except Exception:
			shares_memory = False

		try:
			print(
				"[CALIB] A BOX="
				f"{regions[0].box}; B BOX={regions[1].box}; "
				f"A rect={area_a_view.crop_rect}; "
				f"B rect={area_b_view.crop_rect}; "
				f"A CROP SHAPE={area_a_view.source_crop.shape}; "
				f"B CROP SHAPE={area_b_view.source_crop.shape}; "
				f"A/B shares_memory="
				f"{shares_memory}"
			)
		except Exception:
			pass

	return PlacementRegions(
		region_a=regions[0],
		region_b=regions[1],
		area_a_image=area_a_view.source_crop,
		area_b_image=area_b_view.source_crop,
		pair_score=evaluation.pair_score,
		pair_evaluation=evaluation,
		selected_left_candidate=evaluation.left_candidate,
		selected_right_candidate=evaluation.right_candidate,
		area_a_view=area_a_view,
		area_b_view=area_b_view,
		area_a_warp=area_a_view.warp_image,
		area_b_warp=area_b_view.warp_image,
	)


def build_placement_from_regions(
	frame,
	regions: list[PlacementRegion],
	pair_score: float = 0.0,
	pair_evaluation: PairEvaluation | None = None,
) -> PlacementRegions | None:
	if len(regions) < 2:
		return None

	area_a_view = _build_area_view(
		"A",
		frame,
		regions[0],
	)
	area_b_view = _build_area_view(
		"B",
		frame,
		regions[1],
	)

	return PlacementRegions(
		region_a=regions[0],
		region_b=regions[1],
		area_a_image=area_a_view.source_crop,
		area_b_image=area_b_view.source_crop,
		pair_score=pair_score,
		pair_evaluation=pair_evaluation,
		selected_left_candidate=None,
		selected_right_candidate=None,
		area_a_view=area_a_view,
		area_b_view=area_b_view,
		area_a_warp=area_a_view.warp_image,
		area_b_warp=area_b_view.warp_image,
	)


def placement_uses_selected_pair(placement: PlacementRegions) -> bool:
	def points_close(left_points, right_points) -> bool:
		if len(left_points) != len(right_points):
			return False

		for left_point, right_point in zip(left_points, right_points):
			if abs(float(left_point[0]) - float(right_point[0])) > 0.001:
				return False

			if abs(float(left_point[1]) - float(right_point[1])) > 0.001:
				return False

		return True

	return bool(
		points_close(placement.region_a.box, placement.selected_left_candidate.box)
		and points_close(placement.region_b.box, placement.selected_right_candidate.box)
	)


def _local_to_global(region_box, local_center: tuple[int, int]) -> tuple[float, float]:
	import cv2
	import detector
	import numpy as np

	src = np.array(
		[
			[0, 0],
			[detector.REGION_WARP_WIDTH - 1, 0],
			[detector.REGION_WARP_WIDTH - 1, detector.REGION_WARP_HEIGHT - 1],
			[0, detector.REGION_WARP_HEIGHT - 1],
		],
		dtype=np.float32,
	)
	dst = _order_points(region_box)
	matrix = cv2.getPerspectiveTransform(src, dst)
	point = np.array([[[local_center[0], local_center[1]]]], dtype=np.float32)
	global_point = cv2.perspectiveTransform(point, matrix)[0][0]
	return float(global_point[0]), float(global_point[1])


def _area_local_to_frame(view: AreaView, local_center: tuple[int, int]) -> tuple[float, float]:
	return (
		float(local_center[0] + view.crop_offset[0]),
		float(local_center[1] + view.crop_offset[1]),
	)


def _area_local_to_standard(view: AreaView, local_center: tuple[int, int]) -> tuple[float | None, float | None]:
	if view.homography is None:
		return None, None

	import cv2
	import numpy as np

	frame_x, frame_y = _area_local_to_frame(view, local_center)
	point = np.array([[[frame_x, frame_y]]], dtype=np.float32)
	standard_point = cv2.perspectiveTransform(point, view.homography)[0][0]
	return float(standard_point[0]), float(standard_point[1])


def _normalize_angle_deg(angle_deg: float) -> float:
	normalized = (float(angle_deg) + 180.0) % 360.0 - 180.0

	if normalized == -180.0:
		return 180.0

	return normalized


def _edge_angle_to_mechanical_deg(start, end) -> float:
	import math

	dx = float(end[0]) - float(start[0])
	dy_image = float(end[1]) - float(start[1])
	return _normalize_angle_deg(math.degrees(math.atan2(-dy_image, dx)))


def _standard_contour_points(view: AreaView, contour):
	if view.homography is None:
		return None

	import cv2
	import numpy as np

	points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)

	if len(points) < 3:
		return None

	points[:, 0] += float(view.crop_offset[0])
	points[:, 1] += float(view.crop_offset[1])
	mapped = cv2.perspectiveTransform(points.reshape(-1, 1, 2), view.homography)
	return mapped.reshape(-1, 2)


def _object_axis_angle_from_standard_contour(view: AreaView, detection) -> float | None:
	if detection.shape_name == "ball":
		return None

	if not hasattr(detection, "contour"):
		return None

	standard_points = _standard_contour_points(view, detection.contour)

	if standard_points is None:
		return None

	import cv2
	import numpy as np

	rect = cv2.minAreaRect(standard_points.astype(np.float32).reshape(-1, 1, 2))
	box = cv2.boxPoints(rect)
	edges = []

	for index in range(4):
		start = box[index]
		end = box[(index + 1) % 4]
		length = float(np.linalg.norm(end - start))

		if length > 0:
			edges.append((length, start, end))

	if not edges:
		return None

	_, start, end = max(edges, key=lambda item: item[0])
	return _edge_angle_to_mechanical_deg(start, end)


def _append_area_detections(
	scene_detections: list[SceneDetection],
	view: AreaView,
	detections,
	area_stabilizers: dict | None,
) -> None:
	area = view.name

	if area_stabilizers is not None and area in area_stabilizers:
		detections = area_stabilizers[area].update(detections)

	for detection in detections[:3]:
		global_x, global_y = _area_local_to_frame(view, detection.center)
		standard_x, standard_y = _area_local_to_standard(view, detection.center)
		object_axis_angle_deg = _object_axis_angle_from_standard_contour(
			view,
			detection,
		)
		scene_detections.append(
			SceneDetection(
				shape=detection.shape_name,
				color=detection.color_name,
				center_px=detection.center,
				global_center_x=global_x,
				global_center_y=global_y,
				area=area,
				score=detection.score,
				standard_center_x=standard_x,
				standard_center_y=standard_y,
				object_axis_angle_deg=object_axis_angle_deg,
				is_shadow_cube_fallback=bool(getattr(detection, "is_shadow_cube_fallback", False)),
				is_pink_lab_fallback=bool(getattr(detection, "is_pink_lab_fallback", False)),
			)
		)


def expected_detection_hint(
	task_id: int,
	target_index: int,
	configs: list,
) -> tuple[str | None, str | None]:
	if task_id == protocol.TASK_BASIC_1_2:
		return "cube", None

	if task_id == protocol.TASK_BASIC_1_3:
		return "cube", "pink"

	if task_id == protocol.TASK_ADV_2_1:
		sequence = [
			("pink", "cube"),
			("blue", "cube"),
			("green", "cube"),
		]
		if 0 <= target_index < len(sequence):
			color_name, shape_name = sequence[target_index]
			return shape_name, color_name
		return None, None

	if task_id == protocol.TASK_ADV_2_2:
		if target_index < 0 or target_index >= len(configs):
			return None, None
		return _config_shape_color(configs[target_index])

	if task_id == protocol.TASK_ADV_2_3:
		if target_index < 0 or target_index >= len(configs):
			return None, None
		return _config_shape_color(configs[target_index])

	return None, None


def build_scene_result_from_placement(
	placement: PlacementRegions,
	area_stabilizers: dict | None = None,
	task_id: int | None = None,
	expected_shape: str | None = None,
	expected_color: str | None = None,
) -> SceneBuildResult:
	import detector

	scene_detections = []
	enable_shadow_cube_fallback = (
		config.ENABLE_SHADOW_CUBE_FALLBACK
		and task_id in config.SHADOW_CUBE_FALLBACK_TASK_IDS
		and expected_shape == "cube"
	)

	if placement.area_a_view is None or placement.area_b_view is None:
		return SceneBuildResult([], placement)

	objects_a, _ = detector.detect_all_objects(
		placement.area_a_view.source_crop,
		placement.area_a_view.valid_mask,
		"A",
		enable_shadow_cube_fallback=enable_shadow_cube_fallback,
		task_id=task_id,
		expected_shape=expected_shape,
		expected_color=expected_color,
	)
	objects_b, _ = detector.detect_all_objects(
		placement.area_b_view.source_crop,
		placement.area_b_view.valid_mask,
		"B",
		enable_shadow_cube_fallback=enable_shadow_cube_fallback,
		task_id=task_id,
		expected_shape=expected_shape,
		expected_color=expected_color,
	)
	_append_area_detections(
		scene_detections,
		placement.area_a_view,
		objects_a,
		area_stabilizers,
	)
	_append_area_detections(
		scene_detections,
		placement.area_b_view,
		objects_b,
		area_stabilizers,
	)

	area_a_result = placement.area_a_view.source_crop.copy()
	area_b_result = placement.area_b_view.source_crop.copy()

	for index, detection in enumerate(objects_a[:3], start=1):
		detector.draw_detection(area_a_result, detection, index)

	for index, detection in enumerate(objects_b[:3], start=1):
		detector.draw_detection(area_b_result, detection, index)

	scene_detections.sort(key=lambda item: item.global_center_x)
	return SceneBuildResult(
		detections=scene_detections[:6],
		placement=placement,
		area_a_result=area_a_result,
		area_b_result=area_b_result,
	)


def build_scene_result(
	frame,
	frame_candidates,
	area_stabilizers: dict | None = None,
	task_id: int | None = None,
	expected_shape: str | None = None,
	expected_color: str | None = None,
) -> SceneBuildResult:
	placement = build_placement_regions(frame, frame_candidates)

	if placement is None:
		return SceneBuildResult([], None)

	return build_scene_result_from_placement(
		placement,
		area_stabilizers,
		task_id=task_id,
		expected_shape=expected_shape,
		expected_color=expected_color,
	)


def build_scene_detections(
	frame,
	frame_candidates,
	area_stabilizers: dict | None = None,
	task_id: int | None = None,
	expected_shape: str | None = None,
	expected_color: str | None = None,
) -> list[SceneDetection]:
	return build_scene_result(
		frame,
		frame_candidates,
		area_stabilizers,
		task_id=task_id,
		expected_shape=expected_shape,
		expected_color=expected_color,
	).detections


def _matches_shape(detection: SceneDetection, shape_name: str | None) -> bool:
	return shape_name is None or detection.shape == shape_name


def _matches_color(detection: SceneDetection, color_name: str | None) -> bool:
	return color_name is None or detection.color == color_name


def _config_shape_color(config_item) -> tuple[str | None, str | None]:
	shape_name = protocol.SHAPE_ID_TO_NAME.get(config_item.shape_id)
	color_name = protocol.COLOR_ID_TO_NAME.get(config_item.color_id)

	if color_name == "any":
		color_name = None

	return shape_name, color_name


def _is_completed(detection: SceneDetection, completed_targets: list[tuple[float, float]]) -> bool:
	for done_x, done_y in completed_targets:
		if abs(detection.global_center_x - done_x) <= 35 and abs(detection.global_center_y - done_y) <= 35:
			return True

	return False


def select_target_for_task(
	task_id: int,
	target_index: int,
	detections: list[SceneDetection],
	configs: list,
	completed_targets: list[tuple[float, float]] | None = None,
) -> SceneDetection | None:
	completed_targets = completed_targets or []
	sorted_detections = sorted(detections, key=lambda item: item.global_center_x)

	if task_id == protocol.TASK_BASIC_1_1:
		a_items = [item for item in sorted_detections if item.area == "A"]
		return max(a_items, key=lambda item: item.global_center_x, default=None)

	if task_id == protocol.TASK_BASIC_1_2:
		for item in sorted_detections:
			if item.shape == "cube":
				return item
		return None

	if task_id == protocol.TASK_BASIC_1_3:
		cubes = [
			item
			for item in sorted_detections
			if item.shape == "cube"
		]
		if not cubes:
			return None
		return cubes[0]

	if task_id == protocol.TASK_ADV_2_1:
		sequence = [
			("pink", "cube"),
			("blue", "cube"),
			("green", "cube"),
		]
		if target_index < 0 or target_index >= len(sequence):
			return None
		color_name, shape_name = sequence[target_index]
		for item in sorted_detections:
			if item.color == color_name and item.shape == shape_name:
				return item
		return None

	if task_id == protocol.TASK_ADV_2_2:
		if target_index < 0 or target_index >= len(configs):
			return None
		shape_name, color_name = _config_shape_color(configs[target_index])
		for item in sorted_detections:
			if _matches_shape(item, shape_name) and _matches_color(item, color_name):
				return item
		return None

	if task_id == protocol.TASK_ADV_2_3:
		if target_index < 0 or target_index >= len(configs):
			return None
		shape_name, color_name = _config_shape_color(configs[target_index])
		for item in sorted_detections:
			if _matches_shape(item, shape_name) and _matches_color(item, color_name):
				return item
		return None

	return None


def total_items_for_task(task_id: int, configs: list) -> int:
	if task_id in (protocol.TASK_BASIC_1_1, protocol.TASK_BASIC_1_2):
		return 1

	if task_id == protocol.TASK_BASIC_1_3:
		return 4

	if task_id == protocol.TASK_ADV_2_1:
		return 3

	if task_id in (protocol.TASK_ADV_2_2, protocol.TASK_ADV_2_3):
		return len(configs)

	return 0

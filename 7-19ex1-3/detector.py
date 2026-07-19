from pathlib import Path
import importlib.util
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
draw_detection = _legacy.draw_detection
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


def detect_all_objects(image, valid_mask=None, area_name: str | None = None):
	hsv = _legacy.cv2.cvtColor(image, _legacy.cv2.COLOR_BGR2HSV)
	detections = []
	masks = {}

	for color_name, ranges in _legacy.COLOR_RANGES.items():
		mask = _legacy.build_color_mask(hsv, ranges)
		mask = _apply_valid_mask(mask, valid_mask)
		mask = _clean_task_color_mask(mask)
		masks[color_name] = mask

		contours, _ = _legacy.cv2.findContours(
			mask,
			_legacy.cv2.RETR_EXTERNAL,
			_legacy.cv2.CHAIN_APPROX_SIMPLE,
		)

		for contour in contours:
			area = _legacy.cv2.contourArea(contour)

			if area < _legacy.MIN_OBJECT_AREA:
				continue

			if not _is_valid_task_detection(contour, area, image.shape):
				continue

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

	detections.sort(
		key=lambda item: item.score,
		reverse=True,
	)
	detections = _legacy.remove_duplicate_detections(detections)

	return detections[:6], masks


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

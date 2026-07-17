from pathlib import Path
import importlib.util
import sys


_LEGACY_FILE = Path(__file__).with_name("增加球定点判断版本.py")
_SPEC = importlib.util.spec_from_file_location("legacy_detector", _LEGACY_FILE)

if _SPEC is None or _SPEC.loader is None:
	raise RuntimeError(f"Cannot load detector source: {_LEGACY_FILE}")

_legacy = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _legacy
_SPEC.loader.exec_module(_legacy)


FrameCandidate = _legacy.FrameCandidate
Detection = _legacy.Detection
FrameTracker = _legacy.FrameTracker
ObjectStabilizer = _legacy.ObjectStabilizer

FRAME_UPDATE_INTERVAL = _legacy.FRAME_UPDATE_INTERVAL

COLOR_RANGES = _legacy.COLOR_RANGES
COLOR_TEXT = _legacy.COLOR_TEXT
SHAPE_TEXT = _legacy.SHAPE_TEXT

average_corner_distance = _legacy.average_corner_distance
create_black_mask = _legacy.create_black_mask
find_black_frame_candidates = _legacy.find_black_frame_candidates
warp_frame = _legacy.warp_frame
detect_objects = _legacy.detect_objects
draw_detection = _legacy.draw_detection
draw_frame_candidates = _legacy.draw_frame_candidates
create_mask_preview = _legacy.create_mask_preview


def get_show_frame_candidates() -> bool:
	return bool(_legacy.SHOW_FRAME_CANDIDATES)


def set_show_frame_candidates(value: bool) -> None:
	_legacy.SHOW_FRAME_CANDIDATES = value


def toggle_show_frame_candidates() -> bool:
	_legacy.SHOW_FRAME_CANDIDATES = not _legacy.SHOW_FRAME_CANDIDATES
	return bool(_legacy.SHOW_FRAME_CANDIDATES)

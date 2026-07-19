from dataclasses import dataclass

import cv2

import config

_OPEN_WINDOWS: set[str] = set()


@dataclass
class DisplayState:
	use_warp: bool = True
	show_masks: bool = False
	show_black_mask: bool = False
	should_quit: bool = False
	force_reacquire: bool = False


def print_controls() -> None:
	print("Keys:")
	print("  q: quit")
	print("  c: force black-frame reacquire")
	print("  w: toggle perspective warp")
	print("  m: show/hide color masks")
	print("  b: show/hide black mask")
	print("  f: show/hide all frame candidates")


def show_images(
	source_image,
	result_image,
	masks,
	black_mask,
	state: DisplayState,
) -> None:
	show_window(
		config.WINDOW_SOURCE,
		source_image,
		config.SHOW_SOURCE_CANDIDATES_WINDOW,
	)
	show_window(
		config.WINDOW_RESULT,
		result_image,
		config.SHOW_RESULT_WINDOW,
	)

	show_window(
		config.WINDOW_COLOR_MASKS,
		create_color_mask_preview(masks) if masks else None,
		config.SHOW_COLOR_MASK_WINDOWS and state.show_masks,
	)
	show_window(
		config.WINDOW_BLACK_MASK,
		black_mask,
		config.SHOW_BLACK_MASK_WINDOW and state.show_black_mask,
	)


def show_window(
	name: str,
	image,
	visible: bool,
) -> None:
	if visible:
		if image is None:
			return

		cv2.imshow(
			name,
			image,
		)
		_OPEN_WINDOWS.add(name)
		return

	close_window_if_hidden(name, False)


def read_key(delay_ms: int = 1) -> int:
	return cv2.waitKey(delay_ms) & 0xFF


def create_color_mask_preview(masks):
	import detector

	return detector.create_mask_preview(masks)


def handle_key(key: int, state: DisplayState) -> DisplayState:
	state.force_reacquire = False

	if key == 255:
		return state

	if key == ord("q"):
		state.should_quit = True
		return state

	if key == ord("c"):
		state.force_reacquire = True
		return state

	if key == ord("w"):
		state.use_warp = not state.use_warp
		print(f"Perspective warp: {'on' if state.use_warp else 'off'}")
		return state

	if key == ord("m"):
		state.show_masks = not state.show_masks

		if not state.show_masks:
			close_window_if_hidden(config.WINDOW_COLOR_MASKS, False)

		return state

	if key == ord("b"):
		state.show_black_mask = not state.show_black_mask

		if not state.show_black_mask:
			close_window_if_hidden(config.WINDOW_BLACK_MASK, False)

		return state

	if key == ord("f"):
		config.DEBUG_DRAW_ENABLED = not config.DEBUG_DRAW_ENABLED
		config.DEBUG_DRAW_ALL_CANDIDATES = config.DEBUG_DRAW_ENABLED
		config.DEBUG_DRAW_CANDIDATE_TEXT = config.DEBUG_DRAW_ENABLED
		config.DEBUG_DRAW_ALL_PAIRS = config.DEBUG_DRAW_ENABLED
		config.DEBUG_DRAW_PAIR_TEXT = config.DEBUG_DRAW_ENABLED
		config.DEBUG_DRAW_ROI = config.DEBUG_DRAW_ENABLED
		config.SHOW_SOURCE_CANDIDATES_WINDOW = config.DEBUG_DRAW_ENABLED
		print(f"Frame debug: {'on' if config.DEBUG_DRAW_ENABLED else 'off'}")
		return state

	return state


def close_window_if_hidden(
	name: str,
	visible: bool,
) -> None:
	if visible:
		return

	if name not in _OPEN_WINDOWS:
		return

	close_window(name)
	_OPEN_WINDOWS.discard(name)


def close_window(name: str) -> None:
	try:
		cv2.destroyWindow(name)
	except cv2.error:
		pass


def close_all_windows() -> None:
	cv2.destroyAllWindows()
	_OPEN_WINDOWS.clear()

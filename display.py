from dataclasses import dataclass

import cv2

import config
import detector


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
	cv2.imshow(config.WINDOW_SOURCE, source_image)
	cv2.imshow(config.WINDOW_RESULT, result_image)

	if state.show_masks:
		cv2.imshow(
			config.WINDOW_COLOR_MASKS,
			detector.create_mask_preview(masks),
		)

	if state.show_black_mask and black_mask is not None:
		cv2.imshow(config.WINDOW_BLACK_MASK, black_mask)


def read_key(delay_ms: int = 1) -> int:
	return cv2.waitKey(delay_ms) & 0xFF


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
			close_window(config.WINDOW_COLOR_MASKS)

		return state

	if key == ord("b"):
		state.show_black_mask = not state.show_black_mask

		if not state.show_black_mask:
			close_window(config.WINDOW_BLACK_MASK)

		return state

	if key == ord("f"):
		visible = detector.toggle_show_frame_candidates()
		print(f"Frame candidates: {'on' if visible else 'off'}")
		return state

	return state


def close_window(name: str) -> None:
	try:
		cv2.destroyWindow(name)
	except cv2.error:
		pass


def close_all_windows() -> None:
	cv2.destroyAllWindows()

print("===== VERSION 0720 TEST =====")
import cv2
import numpy as np

import camera
import config
import detector
import display
import serial_comm
import task_state
import vision_task


def _force_reacquire(frame, frame_tracker) -> None:
	candidates, _ = detector.find_black_frame_candidates(frame, None)
	frame_tracker.reset()
	selected = frame_tracker.force_acquire(candidates)

	if selected is None:
		print("No reliable black frame found this time.")
		return

	print(
		"Forced black-frame update: "
		f"score={selected.score:.3f}, "
		f"border={selected.border_cm:.2f}cm, "
		f"ratio={selected.ratio:.2f}"
	)


def _draw_task_overlay(image, controller, task_runner) -> None:
	if not config.DEBUG_DRAW_GLOBAL_OBJECTS:
		return

	lines = [
		f"Task: 0x{controller.current_task:02X}",
		f"State: {controller.state.value}",
		f"Home: {controller.home_count}/{controller.target_count}",
		f"Target: {controller.current_target_index}",
	]

	status = task_runner.status

	if status.active or status.state_text != "idle":
		lines.append(f"Vision: {status.state_text}")

	if status.target_desc:
		lines.append(f"Obj: {status.target_desc}")

	if status.x_cm is not None and status.y_cm is not None:
		lines.append(f"Coord: X {status.x_cm:.2f}cm Y {status.y_cm:.2f}cm")

	if status.base_angle_deg is not None and status.gripper_angle_deg is not None:
		object_axis = (
			"None"
			if status.object_axis_angle_deg is None
			else f"{status.object_axis_angle_deg:.1f}"
		)
		lines.append(
			f"Angle: obj {object_axis} base {status.base_angle_deg:.1f} grip {status.gripper_angle_deg:.1f}"
		)

	if status.message:
		lines.append(status.message[:42])

	for index, text in enumerate(lines):
		cv2.putText(
			image,
			text,
			(10, 112 + index * 24),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.55,
			(0, 255, 255),
			2,
			cv2.LINE_AA,
		)


def _make_waiting_window_image(label: str):
	image = np.zeros(
		(
			config.REGION_WARP_HEIGHT,
			config.REGION_WARP_WIDTH,
			3,
		),
		dtype=np.uint8,
	)
	_, width = image.shape[:2]
	panel_width = min(width, 520)
	cv2.rectangle(
		image,
		(0, 0),
		(panel_width, 52),
		(20, 20, 20),
		-1,
	)
	cv2.putText(
		image,
		label,
		(12, 34),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.7,
		(0, 255, 255),
		2,
		cv2.LINE_AA,
	)
	return image


def _window_image_or_waiting(image, label: str):
	if image is not None:
		return image

	return _make_waiting_window_image(label)


def _show_area_windows(task_runner) -> None:
	display.show_window(
		config.WINDOW_AREA_A,
		_window_image_or_waiting(
			task_runner.area_a_image,
			"area_A waiting for A/B region",
		),
		config.SHOW_AREA_WINDOWS,
	)
	display.show_window(
		config.WINDOW_AREA_B,
		_window_image_or_waiting(
			task_runner.area_b_image,
			"area_B waiting for A/B region",
		),
		config.SHOW_AREA_WINDOWS,
	)
	display.show_window(
		config.WINDOW_AREA_A_RESULT,
		_window_image_or_waiting(
			task_runner.area_a_result,
			"area_A region ready - no object",
		),
		config.SHOW_AREA_RESULT_WINDOWS,
	)
	display.show_window(
		config.WINDOW_AREA_B_RESULT,
		_window_image_or_waiting(
			task_runner.area_b_result,
			"area_B region ready - no object",
		),
		config.SHOW_AREA_RESULT_WINDOWS,
	)
	display.show_window(
		config.WINDOW_AREA_A_WARP,
		_window_image_or_waiting(
			task_runner.area_a_warp,
			"area_A_warp disabled/waiting",
		),
		config.SHOW_AREA_WARP_WINDOWS,
	)
	display.show_window(
		config.WINDOW_AREA_B_WARP,
		_window_image_or_waiting(
			task_runner.area_b_warp,
			"area_B_warp disabled/waiting",
		),
		config.SHOW_AREA_WARP_WINDOWS,
	)


def run() -> None:
	cap = None
	read_failures = 0
	serial_manager = serial_comm.SerialManager()

	frame_tracker = detector.FrameTracker()
	object_stabilizer = detector.ObjectStabilizer()
	task_controller = task_state.VisionTaskController(serial_manager.send)
	task_runner = vision_task.VisionTaskRunner()
	state = display.DisplayState()
	frame_count = 0

	try:
		serial_manager.start()
		cap = camera.open_camera()
		display.print_controls()

		while True:
			for frame_message in serial_manager.get_frames():
				task_controller.handle_frame(frame_message)

			if task_controller.request_vision_reset():
				frame_tracker.reset()
				detector.reset_object_stabilizer(object_stabilizer)
				if task_controller.current_target_index < 0:
					task_runner.cancel()

			task_controller.poll()

			detection_request = task_controller.consume_detection_request()

			if detection_request is not None:
				task_runner.start(detection_request, task_controller.target_count)

			success, frame = camera.read_frame(cap)

			if not success:
				read_failures += 1
				print(
					f"Camera read failed ({read_failures}/{config.READ_FAILURE_LIMIT})."
				)

				if read_failures >= config.READ_FAILURE_LIMIT:
					print("Too many camera read failures, exiting and releasing resources.")
					break

				continue

			read_failures = 0
			frame_count += 1

			display_source = frame.copy()
			candidates = []
			black_mask = None
			area_preview_enabled = (
				config.SHOW_AREA_WINDOWS
				or config.SHOW_AREA_RESULT_WINDOWS
				or config.SHOW_AREA_WARP_WINDOWS
			)

			need_update = (
				frame_tracker.points is None
				or frame_count % detector.FRAME_UPDATE_INTERVAL == 0
			)

			if (
				task_runner.request is not None
			):
				candidates, black_mask = detector.find_black_frame_candidates(
					frame,
					None,
				)

			elif state.use_warp and need_update:
				candidates, black_mask = detector.find_black_frame_candidates(
					frame,
					frame_tracker.points,
				)

				selected, reacquired = frame_tracker.update(candidates)

				if reacquired and selected is not None:
					print(
						"Black frame acquired/switched: "
						f"score={selected.score:.2f}, "
						f"border={selected.border_cm:.2f}cm, "
						f"ratio={selected.ratio:.2f}"
					)

			elif (
				(config.SHOW_BLACK_MASK_WINDOW and state.show_black_mask)
				or config.SHOW_SOURCE_CANDIDATES_WINDOW
				or area_preview_enabled
				or detector.get_show_frame_candidates()
			):
				candidates, black_mask = detector.find_black_frame_candidates(
					frame,
					frame_tracker.points,
				)

			if area_preview_enabled:
				task_runner.update_region_preview(
					frame,
					candidates,
				)

			if task_runner.request is not None:
				task_runner.process_frame(
					frame,
					candidates,
					task_controller,
				)

			if (
				config.SHOW_SOURCE_CANDIDATES_WINDOW
				or config.DEBUG_DRAW_SELECTED_REGIONS
				or config.DEBUG_DRAW_SELECTED_LABELS
			):
				detector.draw_frame_candidates(
					display_source,
					candidates,
					frame_tracker.points,
				)

			if state.use_warp and frame_tracker.points is not None:
				working_image = detector.warp_frame(frame, frame_tracker.points)
				mode_text = "Mode: frame warp"
			else:
				working_image = frame.copy()
				mode_text = "Mode: full image"

			detections, masks = detector.detect_objects(working_image)
			detections = object_stabilizer.update(detections)
			result = working_image.copy()

			if config.DEBUG_DRAW_GLOBAL_OBJECTS:
				for index, detection in enumerate(detections, start=1):
					detector.draw_detection(result, detection, index)

			if config.DEBUG_DRAW_GLOBAL_OBJECTS:
				cv2.putText(
					result,
					mode_text,
					(10, 28),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.7,
					(255, 255, 255),
					2,
					cv2.LINE_AA,
				)
				cv2.putText(
					result,
					f"Objects: {len(detections)}",
					(10, 56),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.7,
					(255, 255, 255),
					2,
					cv2.LINE_AA,
				)

			task_runner.draw_debug(display_source)
			_draw_task_overlay(result, task_controller, task_runner)
			_show_area_windows(task_runner)

			if state.show_black_mask and black_mask is None:
				black_mask = detector.create_black_mask(frame)

			display.show_images(
				display_source,
				result,
				masks,
				black_mask,
				state,
			)

			key = display.read_key()
			state = display.handle_key(key, state)

			if state.force_reacquire:
				_force_reacquire(frame, frame_tracker)

			if state.should_quit:
				break

	finally:
		camera.release_camera(cap)
		serial_manager.close()
		display.close_all_windows()


if __name__ == "__main__":
	run()

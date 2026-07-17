import cv2
import numpy as np

import camera
import config
import detector
import display


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


def run() -> None:
	cap = None
	read_failures = 0

	frame_tracker = detector.FrameTracker()
	object_stabilizer = detector.ObjectStabilizer()
	state = display.DisplayState()
	frame_count = 0

	try:
		cap = camera.open_camera()
		display.print_controls()

		while True:
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

			need_update = (
				frame_tracker.points is None
				or frame_count % detector.FRAME_UPDATE_INTERVAL == 0
			)

			if state.use_warp and need_update:
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

			elif state.show_black_mask or detector.get_show_frame_candidates():
				candidates, black_mask = detector.find_black_frame_candidates(
					frame,
					frame_tracker.points,
				)

			detector.draw_frame_candidates(
				display_source,
				candidates,
				frame_tracker.points,
			)

			if frame_tracker.points is not None:
				cv2.polylines(
					display_source,
					[frame_tracker.points.astype(np.int32)],
					True,
					(255, 255, 255),
					3,
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

			for index, detection in enumerate(detections, start=1):
				detector.draw_detection(result, detection, index)

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

			if frame_tracker.candidate is not None:
				frame_text = (
					f"Frame score={frame_tracker.candidate.score:.2f} "
					f"border={frame_tracker.candidate.border_cm:.2f}cm"
				)
				cv2.putText(
					result,
					frame_text,
					(10, 84),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.58,
					(255, 255, 255),
					2,
					cv2.LINE_AA,
				)

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
		display.close_all_windows()


if __name__ == "__main__":
	run()

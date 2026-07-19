import pathlib
import sys
import importlib
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import serial_protocol as protocol
import task_state
import target_selector
import vision_task
import config
import coordinate
import roi_utils


class FakeClock:
	def __init__(self) -> None:
		self.now = 100.0

	def __call__(self) -> float:
		return self.now

	def advance(self, seconds: float) -> None:
		self.now += seconds


def task_frame(task_id: int) -> protocol.Frame:
	return protocol.parse_frame(bytes([0x01, 0x00, 0x00, 0x00, task_id, 0x10]))


def config_frame(last_flag: int, slot_id: int, shape_id: int, color_id: int) -> protocol.Frame:
	return protocol.parse_frame(bytes([0x02, last_flag, slot_id, shape_id, color_id, 0x20]))


def home_frame() -> protocol.Frame:
	return protocol.parse_frame(protocol.HOME_POSITION_FRAME)


def reset_frame() -> protocol.Frame:
	return protocol.parse_frame(protocol.FORCE_RESET_FRAME)


def det(shape, color, x, y=100, area="B"):
	return target_selector.SceneDetection(
		shape=shape,
		color=color,
		center_px=(int(x), int(y)),
		global_center_x=float(x),
		global_center_y=float(y),
		area=area,
		score=100.0,
		stable_frames=5,
	)


class FakeCandidate:
	def __init__(
		self,
		center_x,
		center_y,
		width,
		height,
		score,
	):
		self.center = (float(center_x), float(center_y))
		self.score = float(score)
		self.ratio = max(float(width), float(height)) / max(1.0, min(float(width), float(height)))
		left = float(center_x) - float(width) / 2.0
		right = float(center_x) + float(width) / 2.0
		top = float(center_y) - float(height) / 2.0
		bottom = float(center_y) + float(height) / 2.0
		self.box = [
			(left, top),
			(right, top),
			(right, bottom),
			(left, bottom),
		]


class ProtocolParserTests(unittest.TestCase):
	def test_home_frame_parser(self):
		parser = protocol.FrameParser()
		frames = parser.feed(protocol.HOME_POSITION_FRAME)
		self.assertEqual(len(frames), 1)
		self.assertEqual(frames[0].raw, protocol.HOME_POSITION_FRAME)

	def test_half_frame_sticky_packet_and_bad_tail_recovery(self):
		parser = protocol.FrameParser()
		self.assertEqual(parser.feed(bytes([0x01, 0x00])), [])
		frames = parser.feed(bytes([0x00, 0x00, 0x11, 0x10]) + protocol.HOME_POSITION_FRAME)
		self.assertEqual([frame.raw for frame in frames], [
			bytes([0x01, 0x00, 0x00, 0x00, 0x11, 0x10]),
			protocol.HOME_POSITION_FRAME,
		])

		parser = protocol.FrameParser()
		frames = parser.feed(bytes([0x01, 0x00, 0x00, 0x00, 0x11, 0x99]) + protocol.FORCE_RESET_FRAME)
		self.assertEqual([frame.raw for frame in frames], [protocol.FORCE_RESET_FRAME])

	def test_old_arm_ack_like_frame_is_ignored(self):
		parser = protocol.FrameParser()
		frames = parser.feed(bytes([0x05, 0x02, 0x11, 0x01, 0x00, 0x50]) + protocol.HOME_POSITION_FRAME)
		self.assertEqual([frame.raw for frame in frames], [protocol.HOME_POSITION_FRAME])

	def test_negative_xy_coordinate_pack(self):
		frame = protocol.build_coord_frame_from_xy(-3.30, -4.10)
		self.assertEqual(frame, bytes([0x03, 0xFE, 0xB6, 0xFE, 0x66, 0x30]))

	def test_positive_xy_coordinate_pack(self):
		frame = protocol.build_coord_frame_from_xy(3.30, 4.10)
		self.assertEqual(frame, bytes([0x03, 0x01, 0x4A, 0x01, 0x9A, 0x30]))


class StateMachineTests(unittest.TestCase):
	def make_controller(self):
		sent = []
		clock = FakeClock()
		controller = task_state.VisionTaskController(sent.append, clock=clock)
		return controller, sent, clock

	def trigger_home(self, controller, clock, seconds=0.51):
		clock.advance(seconds)
		controller.handle_frame(home_frame())

	def test_task_select_has_no_tx(self):
		controller, sent, _ = self.make_controller()
		controller.handle_frame(task_frame(protocol.TASK_BASIC_1_1))
		self.assertEqual(sent, [])
		self.assertEqual(controller.current_task, protocol.TASK_BASIC_1_1)
		self.assertEqual(controller.home_count, 1)
		self.assertEqual(controller.target_count, 1)
		self.assertEqual(controller.current_target_index, 0)
		self.assertEqual(controller.state, task_state.TaskState.READY_TO_DETECT)

	def test_config_has_no_tx(self):
		controller, sent, _ = self.make_controller()
		controller.handle_frame(task_frame(protocol.TASK_ADV_2_2))
		controller.handle_frame(config_frame(0x01, 1, protocol.SHAPE_CUBE, protocol.COLOR_ANY))
		self.assertEqual(sent, [])
		self.assertEqual(controller.target_count, 2)
		self.assertEqual(controller.home_count, 1)
		self.assertEqual(controller.current_target_index, 0)
		self.assertEqual(controller.state, task_state.TaskState.READY_TO_DETECT)

	def test_home_debounce_counts_three_frames_once(self):
		controller, _, clock = self.make_controller()
		controller.handle_frame(task_frame(protocol.TASK_BASIC_1_3))
		clock.advance(0.02)
		controller.handle_frame(home_frame())
		clock.advance(0.02)
		controller.handle_frame(home_frame())
		self.assertEqual(controller.home_count, 1)
		self.assertEqual(controller.current_target_index, 0)

	def test_home_after_debounce_counts_again(self):
		controller, _, clock = self.make_controller()
		controller.handle_frame(task_frame(protocol.TASK_BASIC_1_3))
		controller.mark_detection_started()
		self.trigger_home(controller, clock)
		self.assertEqual(controller.home_count, 2)
		self.assertEqual(controller.current_target_index, 1)

	def test_basic_1_1_second_home_exits(self):
		controller, sent, clock = self.make_controller()
		controller.handle_frame(task_frame(protocol.TASK_BASIC_1_1))
		self.assertEqual(controller.state, task_state.TaskState.READY_TO_DETECT)
		controller.mark_detection_started()
		self.trigger_home(controller, clock)
		self.assertEqual(controller.state, task_state.TaskState.SELECT_MODE)
		self.assertEqual(controller.current_task, 0)
		self.assertEqual(sent, [])

	def test_basic_1_3_fifth_home_exits(self):
		controller, sent, clock = self.make_controller()
		controller.handle_frame(task_frame(protocol.TASK_BASIC_1_3))
		self.assertEqual(controller.current_target_index, 0)
		for index in range(4):
			self.assertEqual(controller.current_target_index, index)
			controller.mark_detection_started()
			if index < 3:
				self.trigger_home(controller, clock)
		self.trigger_home(controller, clock)
		self.assertEqual(controller.state, task_state.TaskState.SELECT_MODE)
		self.assertEqual(controller.home_count, 0)
		self.assertEqual(sent, [])

	def test_advanced_2_1_fourth_home_exits(self):
		controller, _, clock = self.make_controller()
		controller.handle_frame(task_frame(protocol.TASK_ADV_2_1))
		self.assertEqual(controller.current_target_index, 0)
		for index in range(3):
			controller.mark_detection_started()
			self.trigger_home(controller, clock)
			if index < 2:
				self.assertEqual(controller.current_target_index, index + 1)
		self.trigger_home(controller, clock)
		self.assertEqual(controller.state, task_state.TaskState.SELECT_MODE)

	def test_advanced_2_2_third_home_exits(self):
		controller, _, clock = self.make_controller()
		controller.handle_frame(task_frame(protocol.TASK_ADV_2_2))
		controller.handle_frame(config_frame(0x01, 1, protocol.SHAPE_BALL, protocol.COLOR_ANY))
		self.assertEqual(controller.current_target_index, 0)
		for index in range(2):
			controller.mark_detection_started()
			self.trigger_home(controller, clock)
			if index == 0:
				self.assertEqual(controller.current_target_index, 1)
		self.trigger_home(controller, clock)
		self.assertEqual(controller.state, task_state.TaskState.SELECT_MODE)

	def test_advanced_2_3_exits_after_config_count_plus_one(self):
		controller, _, clock = self.make_controller()
		controller.handle_frame(task_frame(protocol.TASK_ADV_2_3))
		controller.handle_frame(config_frame(0x00, 1, protocol.SHAPE_CUBE, protocol.COLOR_PINK))
		controller.handle_frame(config_frame(0x00, 2, protocol.SHAPE_BALL, protocol.COLOR_BLUE))
		controller.handle_frame(config_frame(0x01, 3, protocol.SHAPE_CUBOID, protocol.COLOR_GREEN))
		self.assertEqual(controller.target_count, 3)
		self.assertEqual(controller.current_target_index, 0)

		for index in range(3):
			controller.mark_detection_started()
			self.trigger_home(controller, clock)
			if index < 2:
				self.assertEqual(controller.current_target_index, index + 1)
		self.trigger_home(controller, clock)
		self.assertEqual(controller.state, task_state.TaskState.SELECT_MODE)

	def test_force_reset_clears_counts_without_tx(self):
		controller, sent, _ = self.make_controller()
		controller.handle_frame(task_frame(protocol.TASK_ADV_2_3))
		controller.handle_frame(config_frame(0x00, 1, protocol.SHAPE_CUBE, protocol.COLOR_PINK))
		controller.handle_frame(reset_frame())
		self.assertEqual(controller.current_task, 0)
		self.assertEqual(controller.home_count, 0)
		self.assertEqual(controller.target_count, 0)
		self.assertEqual(controller.current_target_index, -1)
		self.assertEqual(controller.task_config_queue, [])
		self.assertEqual(sent, [])

	def test_new_task_resets_home_count(self):
		controller, _, _ = self.make_controller()
		controller.handle_frame(task_frame(protocol.TASK_BASIC_1_3))
		self.assertEqual(controller.home_count, 1)
		controller.handle_frame(task_frame(protocol.TASK_BASIC_1_1))
		self.assertEqual(controller.home_count, 1)
		self.assertEqual(controller.current_target_index, 0)

	def test_send_coordinate_once_without_ack_wait(self):
		controller, sent, _ = self.make_controller()
		controller.handle_frame(task_frame(protocol.TASK_BASIC_1_1))
		frame = controller.send_target_coordinate(-3.30, -4.10)
		self.assertEqual(sent, [frame])
		self.assertEqual(frame, bytes([0x03, 0xFE, 0xB6, 0xFE, 0x66, 0x30]))
		self.assertEqual(controller.state, task_state.TaskState.WAIT_NEXT_HOME)
		controller.poll()
		self.assertEqual(sent, [frame])


class TargetSelectionTests(unittest.TestCase):
	def test_a_area_rightmost_target(self):
		items = [
			det("cube", "pink", 20, area="B"),
			det("ball", "green", 80, area="A"),
			det("cube", "blue", 120, area="A"),
		]
		target = target_selector.select_target_for_task(protocol.TASK_BASIC_1_1, 0, items, [])
		self.assertEqual(target.global_center_x, 120)

	def test_global_leftmost_first_cube(self):
		items = [
			det("ball", "pink", 10),
			det("cube", "green", 30),
			det("cube", "blue", 90),
		]
		target = target_selector.select_target_for_task(protocol.TASK_BASIC_1_2, 0, items, [])
		self.assertEqual(target.global_center_x, 30)

	def test_max_six_sorted(self):
		items = [det("cube", "pink", x) for x in [60, 10, 50, 20, 40, 30, 70]]
		items.sort(key=lambda item: item.global_center_x)
		self.assertEqual([item.global_center_x for item in items[:6]], [10, 20, 30, 40, 50, 60])

	def test_basic_1_3_uses_current_leftmost_without_old_coordinate_filter(self):
		first_view = [
			det("cube", "pink", 10),
			det("cube", "blue", 70),
			det("cube", "green", 130),
			det("cube", "orange", 190),
		]
		target = target_selector.select_target_for_task(protocol.TASK_BASIC_1_3, 0, first_view, [])
		self.assertEqual(target.global_center_x, 10)

		second_view = [
			det("cube", "blue", 20),
			det("cube", "green", 80),
			det("cube", "orange", 140),
		]
		target = target_selector.select_target_for_task(
			protocol.TASK_BASIC_1_3,
			1,
			second_view,
			[],
			completed_targets=[(10, 100)],
		)
		self.assertEqual(target.global_center_x, 20)

	def test_basic_1_3_close_neighbor_is_not_filtered(self):
		items = [
			det("cube", "pink", 10),
			det("cube", "blue", 30),
		]
		target = target_selector.select_target_for_task(
			protocol.TASK_BASIC_1_3,
			1,
			items,
			[],
			completed_targets=[(10, 100)],
		)
		self.assertEqual(target.global_center_x, 10)

	def test_advanced_2_1_color_order(self):
		items = [
			det("cube", "green", 30),
			det("cube", "pink", 10),
			det("cube", "blue", 20),
		]
		self.assertEqual(target_selector.select_target_for_task(protocol.TASK_ADV_2_1, 0, items, []).color, "pink")
		self.assertEqual(target_selector.select_target_for_task(protocol.TASK_ADV_2_1, 1, items, []).color, "blue")
		self.assertEqual(target_selector.select_target_for_task(protocol.TASK_ADV_2_1, 2, items, []).color, "green")

	def test_advanced_2_2_any_color_shape(self):
		configs = [task_state.TargetConfig(slot_id=7, shape_id=protocol.SHAPE_BALL, color_id=protocol.COLOR_ANY)]
		items = [
			det("ball", "pink", 10),
			det("cube", "blue", 15),
			det("ball", "green", 20),
		]
		target = target_selector.select_target_for_task(protocol.TASK_ADV_2_2, 1, items, configs)
		self.assertEqual(target.global_center_x, 10)

	def test_advanced_2_2_reselects_current_leftmost_after_first_removed(self):
		configs = [task_state.TargetConfig(slot_id=7, shape_id=protocol.SHAPE_BALL, color_id=protocol.COLOR_ANY)]
		first_view = [
			det("ball", "pink", 10),
			det("ball", "green", 40),
		]
		first_target = target_selector.select_target_for_task(protocol.TASK_ADV_2_2, 0, first_view, configs)
		self.assertEqual(first_target.global_center_x, 10)

		second_view = [
			det("ball", "green", 40),
		]
		second_target = target_selector.select_target_for_task(protocol.TASK_ADV_2_2, 1, second_view, configs)
		self.assertEqual(second_target.global_center_x, 40)

	def test_advanced_2_2_more_than_two_matches_still_uses_current_leftmost(self):
		configs = [task_state.TargetConfig(slot_id=7, shape_id=protocol.SHAPE_CUBE, color_id=protocol.COLOR_ANY)]
		items = [
			det("cube", "green", 80),
			det("cube", "pink", 15),
			det("cube", "blue", 45),
		]
		target = target_selector.select_target_for_task(protocol.TASK_ADV_2_2, 1, items, configs)
		self.assertEqual(target.global_center_x, 15)

	def test_advanced_2_3_config_queue(self):
		configs = [
			task_state.TargetConfig(slot_id=1, shape_id=protocol.SHAPE_CUBE, color_id=protocol.COLOR_PINK),
			task_state.TargetConfig(slot_id=6, shape_id=protocol.SHAPE_BALL, color_id=protocol.COLOR_BLUE),
		]
		items = [
			det("ball", "blue", 20),
			det("cube", "pink", 10),
		]
		self.assertEqual(target_selector.select_target_for_task(protocol.TASK_ADV_2_3, 0, items, configs).color, "pink")
		self.assertEqual(target_selector.select_target_for_task(protocol.TASK_ADV_2_3, 1, items, configs).shape, "ball")


class ShadowCubeFallbackTests(unittest.TestCase):
	def test_shadow_cube_config_includes_allowed_cube_tasks(self):
		self.assertTrue(config.ENABLE_SHADOW_CUBE_FALLBACK)
		self.assertEqual(
			config.SHADOW_CUBE_FALLBACK_TASK_IDS,
			{
				protocol.TASK_BASIC_1_3,
				protocol.TASK_ADV_2_2,
				protocol.TASK_ADV_2_3,
			},
		)

	def test_task_id_is_passed_without_global_task_state(self):
		source = (ROOT / "vision_task.py").read_text(encoding="utf-8")
		self.assertIn("task_id=self.request.task_id", source)
		self.assertIn("expected_shape, expected_color", source)
		source = (ROOT / "target_selector.py").read_text(encoding="utf-8")
		self.assertIn("task_id: int | None = None", source)
		self.assertIn("enable_shadow_cube_fallback=enable_shadow_cube_fallback", source)
		self.assertIn("expected_shape: str | None = None", source)
		self.assertIn("expected_color: str | None = None", source)

	def test_expected_detection_hint_marks_basic_1_3_as_pink_cube(self):
		self.assertEqual(
			target_selector.expected_detection_hint(protocol.TASK_BASIC_1_3, 0, []),
			("cube", "pink"),
		)
		self.assertEqual(
			target_selector.expected_detection_hint(protocol.TASK_BASIC_1_1, 0, []),
			(None, None),
		)

	def test_target_selector_enables_shadow_fallback_only_for_basic_1_3(self):
		class FakeImage:
			def __init__(self, label):
				self.label = label

			def copy(self):
				return FakeImage(f"{self.label}_copy")

		class FakeDetection:
			def __init__(self):
				self.shape_name = "cube"
				self.color_name = "green"
				self.center = (20, 20)
				self.score = 1.0

		calls = []

		def fake_detect_all_objects(
			image,
			valid_mask=None,
			area_name=None,
			enable_shadow_cube_fallback=False,
			task_id=None,
			expected_shape=None,
			expected_color=None,
		):
			calls.append((area_name, enable_shadow_cube_fallback, task_id, expected_shape, expected_color))
			return [FakeDetection()], {}

		fake_detector = types.SimpleNamespace(
			detect_all_objects=fake_detect_all_objects,
			draw_detection=lambda image, detection, index: None,
		)
		old_detector = sys.modules.get("detector")
		sys.modules["detector"] = fake_detector

		def make_placement():
			area_a_region = target_selector.PlacementRegion(
				box=[],
				score=1.0,
				center=(10.0, 10.0),
				area="A",
			)
			area_b_region = target_selector.PlacementRegion(
				box=[],
				score=1.0,
				center=(30.0, 10.0),
				area="B",
			)
			return target_selector.PlacementRegions(
				region_a=area_a_region,
				region_b=area_b_region,
				area_a_image=FakeImage("A"),
				area_b_image=FakeImage("B"),
				pair_score=1.0,
				pair_evaluation=None,
				selected_left_candidate=None,
				selected_right_candidate=None,
				area_a_view=target_selector.AreaView(
					name="A",
					region=area_a_region,
					source_crop=FakeImage("A"),
					valid_mask=None,
					result_image=FakeImage("A_result"),
					crop_offset=(0, 0),
				),
				area_b_view=target_selector.AreaView(
					name="B",
					region=area_b_region,
					source_crop=FakeImage("B"),
					valid_mask=None,
					result_image=FakeImage("B_result"),
					crop_offset=(100, 0),
				),
			)

		try:
			target_selector.build_scene_result_from_placement(
				make_placement(),
				task_id=protocol.TASK_BASIC_1_3,
				expected_shape="cube",
			)
			self.assertEqual(
				calls,
				[
					("A", True, protocol.TASK_BASIC_1_3, "cube", None),
					("B", True, protocol.TASK_BASIC_1_3, "cube", None),
				],
			)
			calls.clear()
			target_selector.build_scene_result_from_placement(
				make_placement(),
				task_id=protocol.TASK_ADV_2_2,
				expected_shape="cube",
			)
			self.assertEqual(
				calls,
				[
					("A", True, protocol.TASK_ADV_2_2, "cube", None),
					("B", True, protocol.TASK_ADV_2_2, "cube", None),
				],
			)
			calls.clear()
			target_selector.build_scene_result_from_placement(
				make_placement(),
				task_id=protocol.TASK_ADV_2_3,
				expected_shape="cuboid",
			)
			self.assertEqual(
				calls,
				[
					("A", False, protocol.TASK_ADV_2_3, "cuboid", None),
					("B", False, protocol.TASK_ADV_2_3, "cuboid", None),
				],
			)
		finally:
			if old_detector is None:
				del sys.modules["detector"]
			else:
				sys.modules["detector"] = old_detector

	def test_shadow_cube_fallback_recovers_core_center_when_cv2_available(self):
		try:
			import cv2
			import numpy as np
			import detector
		except ModuleNotFoundError:
			self.skipTest("cv2/numpy is not installed in this Python environment")

		image = np.zeros((140, 220, 3), dtype=np.uint8)
		image[:] = (255, 255, 255)
		cv2.rectangle(image, (12, 40), (70, 98), (0, 255, 0), -1)
		cv2.rectangle(image, (70, 58), (155, 78), (0, 85, 0), -1)

		detections, _ = detector.detect_all_objects(
			image,
			area_name="A",
			enable_shadow_cube_fallback=True,
		)
		fallbacks = [
			item
			for item in detections
			if getattr(item, "is_shadow_cube_fallback", False)
		]
		self.assertTrue(fallbacks)
		self.assertEqual(fallbacks[0].shape_name, "cube")
		self.assertLess(fallbacks[0].center[0], 75)

	def test_normal_cube_is_not_marked_as_shadow_fallback_when_cv2_available(self):
		try:
			import cv2
			import numpy as np
			import detector
		except ModuleNotFoundError:
			self.skipTest("cv2/numpy is not installed in this Python environment")

		image = np.zeros((140, 220, 3), dtype=np.uint8)
		image[:] = (255, 255, 255)
		cv2.rectangle(image, (12, 40), (70, 98), (0, 255, 0), -1)

		detections, _ = detector.detect_all_objects(
			image,
			area_name="A",
			enable_shadow_cube_fallback=True,
		)
		cubes = [item for item in detections if item.shape_name == "cube"]
		self.assertTrue(cubes)
		self.assertFalse(any(getattr(item, "is_shadow_cube_fallback", False) for item in cubes))

	def test_middle_candidate_does_not_use_shadow_fallback_when_cv2_available(self):
		try:
			import cv2
			import numpy as np
			import detector
		except ModuleNotFoundError:
			self.skipTest("cv2/numpy is not installed in this Python environment")

		image = np.zeros((140, 220, 3), dtype=np.uint8)
		image[:] = (255, 255, 255)
		cv2.rectangle(image, (82, 40), (140, 98), (0, 255, 0), -1)
		cv2.rectangle(image, (140, 58), (205, 78), (0, 85, 0), -1)

		detections, _ = detector.detect_all_objects(
			image,
			area_name="A",
			enable_shadow_cube_fallback=True,
		)
		self.assertFalse(any(getattr(item, "is_shadow_cube_fallback", False) for item in detections))


class PinkLabFallbackTests(unittest.TestCase):
	def make_pink_lab_image(self, area_name="A", shape_name="cube"):
		try:
			import cv2
			import numpy as np
		except ModuleNotFoundError:
			self.skipTest("cv2/numpy is not installed in this Python environment")

		image = np.full((140, 220, 3), 255, dtype=np.uint8)
		color = (230, 218, 255)

		if area_name == "A":
			if shape_name == "cube":
				cv2.rectangle(image, (12, 42), (64, 94), color, -1)
			elif shape_name == "cuboid":
				cv2.rectangle(image, (12, 48), (116, 88), color, -1)
			elif shape_name == "ball":
				cv2.circle(image, (42, 70), 28, color, -1)
			elif shape_name == "middle":
				cv2.rectangle(image, (92, 42), (144, 94), color, -1)
		else:
			cv2.rectangle(image, (156, 42), (208, 94), color, -1)

		return image, np.zeros((140, 220), dtype=np.uint8)

	def detect_pink_lab_direct(self, image, pink_mask, area_name, task_id=protocol.TASK_BASIC_1_3):
		import detector

		return detector._detect_pink_cube_with_lab_fallback(
			image,
			None,
			area_name,
			pink_mask,
			[],
			task_id,
			"cube",
			"pink",
		)

	def test_pink_lab_config_is_limited_to_basic_1_3(self):
		self.assertTrue(config.ENABLE_PINK_LAB_CUBE_FALLBACK)
		self.assertEqual(
			config.PINK_LAB_FALLBACK_TASK_IDS,
			{
				protocol.TASK_BASIC_1_3,
				protocol.TASK_ADV_2_2,
				protocol.TASK_ADV_2_3,
			},
		)

	def test_a_left_pink_cube_lab_fallback_when_cv2_available(self):
		image, pink_mask = self.make_pink_lab_image("A", "cube")
		detection = self.detect_pink_lab_direct(image, pink_mask, "A")
		self.assertIsNotNone(detection)
		self.assertTrue(getattr(detection, "is_pink_lab_fallback", False))
		self.assertEqual(detection.shape_name, "cube")
		self.assertEqual(detection.color_name, "pink")
		self.assertLess(detection.center[0], 80)

	def test_b_right_pink_cube_lab_fallback_when_cv2_available(self):
		image, pink_mask = self.make_pink_lab_image("B", "cube")
		detection = self.detect_pink_lab_direct(image, pink_mask, "B")
		self.assertIsNotNone(detection)
		self.assertTrue(getattr(detection, "is_pink_lab_fallback", False))
		self.assertGreater(detection.center[0], 145)

	def test_pink_lab_rejects_cuboid_ball_middle_and_other_task_when_cv2_available(self):
		for shape_name in ("cuboid", "ball", "middle"):
			image, pink_mask = self.make_pink_lab_image("A", shape_name)
			self.assertIsNone(self.detect_pink_lab_direct(image, pink_mask, "A"))

		image, pink_mask = self.make_pink_lab_image("A", "cube")
		self.assertIsNone(
			self.detect_pink_lab_direct(
				image,
				pink_mask,
				"A",
				task_id=protocol.TASK_BASIC_1_1,
			)
		)

	def test_pink_lab_does_not_run_without_expected_pink_cube_when_cv2_available(self):
		image, pink_mask = self.make_pink_lab_image("A", "cube")
		import detector

		self.assertIsNone(
			detector._detect_pink_cube_with_lab_fallback(
				image,
				None,
				"A",
				pink_mask,
				[],
				protocol.TASK_BASIC_1_3,
				"cube",
				"green",
			)
		)


class CoordinateSendTests(unittest.TestCase):
	def make_controller(self):
		sent = []
		clock = FakeClock()
		controller = task_state.VisionTaskController(sent.append, clock=clock)
		controller.handle_frame(task_frame(protocol.TASK_BASIC_1_1))
		return controller, sent

	def make_ideal_placement(self):
		area_a_region = target_selector.PlacementRegion(
			box=[
				(0.0, 0.0),
				(170.0, 0.0),
				(170.0, 130.0),
				(0.0, 130.0),
			],
			score=1.0,
			center=(85.0, 65.0),
			area="A",
		)
		area_b_region = target_selector.PlacementRegion(
			box=[
				(170.0, 0.0),
				(340.0, 0.0),
				(340.0, 130.0),
				(170.0, 130.0),
			],
			score=1.0,
			center=(255.0, 65.0),
			area="B",
		)
		return target_selector.PlacementRegions(
			region_a=area_a_region,
			region_b=area_b_region,
			area_a_image=None,
			area_b_image=None,
			pair_score=1.0,
			pair_evaluation=None,
			selected_left_candidate=None,
			selected_right_candidate=None,
		)

	def test_missing_stable_placement_runner_does_not_send_coordinate(self):
		controller, sent = self.make_controller()
		runner = vision_task.VisionTaskRunner()
		target = det("cube", "pink", 50, 60, "A")
		runner.target_history = [target for _ in range(5)]
		runner.request = task_state.DetectionRequest(protocol.TASK_BASIC_1_1, 0, 1, tuple())
		runner._send_stable_target(target, controller)
		self.assertEqual(sent, [])
		self.assertEqual(controller.state, task_state.TaskState.ERROR)
		self.assertEqual(controller.current_target_index, 0)
		self.assertIsNotNone(runner.request)

	def test_missing_placement_home_does_not_advance_and_retry_sends_same_target(self):
		controller, sent = self.make_controller()
		runner = vision_task.VisionTaskRunner()
		target = det("cube", "pink", 50, 60, "A")
		runner.target_history = [target for _ in range(5)]
		runner.request = task_state.DetectionRequest(protocol.TASK_BASIC_1_1, 0, 1, tuple())
		runner._send_stable_target(target, controller)
		controller.handle_frame(home_frame())
		self.assertEqual(controller.home_count, 1)
		self.assertEqual(controller.current_target_index, 0)
		self.assertEqual(sent, [])

		old_converter = coordinate.frame_point_to_arm_coordinate
		runner.stable_placement = object()
		coordinate.frame_point_to_arm_coordinate = lambda frame_x, frame_y, placement: coordinate.CoordinateResult(
			x_cm=-3.30,
			y_cm=-4.10,
			x_raw=-330,
			y_raw=-410,
		)
		try:
			runner._send_stable_target(target, controller)
			self.assertEqual(len(sent), 1)
			self.assertEqual(sent[0], bytes([0x03, 0xFE, 0xB6, 0xFE, 0x66, 0x30]))
			self.assertEqual(controller.state, task_state.TaskState.WAIT_NEXT_HOME)
			self.assertEqual(controller.current_target_index, 0)
		finally:
			coordinate.frame_point_to_arm_coordinate = old_converter

	def test_force_reset_after_calibration_error_clears_state(self):
		controller, _ = self.make_controller()
		controller.state = task_state.TaskState.ERROR
		controller.handle_frame(reset_frame())
		self.assertEqual(controller.current_task, 0)
		self.assertEqual(controller.home_count, 0)
		self.assertEqual(controller.current_target_index, -1)

	def test_send_coordinate_enters_wait_next_home(self):
		controller, sent = self.make_controller()
		frame = controller.send_target_coordinate(-3.30, -4.10)
		self.assertEqual(sent, [frame])
		self.assertEqual(frame, bytes([0x03, 0xFE, 0xB6, 0xFE, 0x66, 0x30]))
		self.assertEqual(controller.state, task_state.TaskState.WAIT_NEXT_HOME)

	def test_ideal_unified_ab_plane_maps_to_arm_xy(self):
		try:
			import cv2
			import numpy
		except ModuleNotFoundError:
			self.skipTest("cv2 or numpy is not installed in this Python environment")

		_ = cv2, numpy
		placement = self.make_ideal_placement()
		area_a_center = coordinate.frame_point_to_arm_coordinate(85.0, 65.0, placement)
		area_b_center = coordinate.frame_point_to_arm_coordinate(255.0, 65.0, placement)
		mid_top = coordinate.frame_point_to_arm_coordinate(170.0, 0.0, placement)
		self.assertAlmostEqual(area_a_center.x_cm, -8.5, places=2)
		self.assertAlmostEqual(area_a_center.y_cm, -14.0, places=2)
		self.assertAlmostEqual(area_b_center.x_cm, 8.5, places=2)
		self.assertAlmostEqual(area_b_center.y_cm, -14.0, places=2)
		self.assertAlmostEqual(mid_top.x_cm, 0.0, places=2)
		self.assertAlmostEqual(mid_top.y_cm, -7.5, places=2)


class BlackFrameSelectionTests(unittest.TestCase):
	def test_two_inner_frames_pair_to_two_regions(self):
		regions = target_selector.select_two_regions([
			FakeCandidate(200, 100, 170, 130, 0.8),
			FakeCandidate(390, 102, 168, 132, 0.78),
		])
		self.assertEqual(len(regions), 2)
		self.assertLess(regions[0].center[0], regions[1].center[0])

	def test_left_region_is_a_and_right_region_is_b(self):
		regions = target_selector.select_two_regions([
			FakeCandidate(200, 100, 170, 130, 0.8),
			FakeCandidate(390, 100, 170, 130, 0.8),
		])
		self.assertEqual(regions[0].area, "A")
		self.assertEqual(regions[1].area, "B")

	def test_best_pair_ignores_unpaired_high_score_candidate(self):
		candidates = [
			FakeCandidate(650, 100, 170, 130, 0.95),
			FakeCandidate(200, 100, 202, 130, 0.30),
			FakeCandidate(430, 101, 204, 130, 0.30),
		]
		regions = target_selector.select_two_regions(candidates)
		self.assertEqual(len(regions), 2)
		self.assertAlmostEqual(regions[0].center[0], 200.0)

	def test_region_centers_are_original_candidate_centers(self):
		regions = target_selector.select_two_regions([
			FakeCandidate(200, 100, 170, 130, 0.8),
			FakeCandidate(390, 100, 170, 130, 0.8),
		])
		self.assertAlmostEqual(regions[0].center[0], 200.0)
		self.assertAlmostEqual(regions[1].center[0], 390.0)

	def test_no_candidate_returns_no_regions(self):
		self.assertEqual(target_selector.select_two_regions([]), [])

	def test_single_inner_candidate_returns_no_regions(self):
		regions = target_selector.select_two_regions([
			FakeCandidate(200, 100, 170, 130, 0.8),
		])
		self.assertEqual(regions, [])

	def test_bad_ratio_candidate_does_not_pair(self):
		candidates = [
			FakeCandidate(200, 100, 300, 80, 0.8),
			FakeCandidate(390, 100, 170, 130, 0.8),
		]
		regions = target_selector.select_two_regions(candidates)
		self.assertEqual(regions, [])

	def test_roi_bounds_default_lower_area(self):
		x0, y0, x1, y1 = roi_utils.get_placement_roi_bounds(1280, 720)
		self.assertEqual((x0, x1), (0, 1280))
		self.assertEqual(y0, round(720 * config.PLACEMENT_ROI_Y_MIN_RATIO))
		self.assertEqual(y1, 720)

	def test_roi_offset_point_to_global(self):
		self.assertEqual(
			roi_utils.offset_point_to_global(12.5, 30.0, 100, 200),
			(112.5, 230.0),
		)

	def test_inner_region_ratio_score_prefers_17_to_13(self):
		good = FakeCandidate(200, 100, 170, 130, 0.8)
		bad = FakeCandidate(390, 100, 260, 100, 0.8)
		self.assertGreater(
			target_selector._ratio_score(good),
			target_selector._ratio_score(bad),
		)

	def test_field_seen_ratio_around_156_can_pair(self):
		candidates = [
			FakeCandidate(200, 100, 202, 130, 0.24),
			FakeCandidate(430, 102, 204, 130, 0.24),
		]
		regions = target_selector.select_two_regions(candidates)
		self.assertEqual(len(regions), 2)
		self.assertEqual(regions[0].area, "A")
		self.assertEqual(regions[1].area, "B")

	def test_small_negative_gap_can_pair(self):
		candidates = [
			FakeCandidate(200, 100, 170, 130, 0.8),
			FakeCandidate(360, 100, 170, 130, 0.8),
		]
		evaluation = target_selector.select_inner_region_pair(candidates)
		self.assertIsNotNone(evaluation)
		self.assertTrue(evaluation.accepted)
		self.assertLess(evaluation.gap_ratio, 0.0)

	def test_pair_evaluation_reports_reject_reason(self):
		candidates = [
			FakeCandidate(200, 100, 170, 130, 0.8),
			FakeCandidate(900, 100, 170, 130, 0.8),
		]
		evaluations = target_selector.evaluate_candidate_pairs(candidates)
		self.assertEqual(evaluations[0].reject_reason, "gap_too_large")

	def test_build_placement_regions_returns_area_images(self):
		class FakeImage:
			def __init__(self, label, shape):
				self.label = label
				self.shape = shape

			def copy(self):
				return FakeImage(f"{self.label}_copy", self.shape)

		def fake_build_area_view(name, frame, region):
			image = FakeImage(name, (120, 160, 3))
			return target_selector.AreaView(
				name=name,
				region=region,
				source_crop=image,
				valid_mask=f"mask_{name}",
				result_image=image.copy(),
				crop_offset=(10 if name == "A" else 220, 30),
				warp_image=FakeImage(f"{name}_warp", (config.REGION_WARP_HEIGHT, config.REGION_WARP_WIDTH, 3)),
			)

		old_build_area_view = target_selector._build_area_view
		target_selector._build_area_view = fake_build_area_view
		try:
			placement = target_selector.build_placement_regions(
				object(),
				[
					FakeCandidate(200, 100, 170, 130, 0.8),
					FakeCandidate(390, 100, 170, 130, 0.8),
				],
			)
			self.assertIsNotNone(placement)
			self.assertEqual(placement.area_a_image.label, "A")
			self.assertEqual(placement.area_b_image.label, "B")
			self.assertNotEqual(placement.area_a_image.label, placement.area_b_image.label)
			self.assertEqual(placement.area_a_warp.label, "A_warp")
			self.assertEqual(placement.area_b_warp.label, "B_warp")
		finally:
			target_selector._build_area_view = old_build_area_view

	def test_build_placement_regions_warps_selected_pair_not_top_single_candidate(self):
		class FakeImage:
			def __init__(self, label):
				self.label = label
				self.shape = (config.REGION_WARP_HEIGHT, config.REGION_WARP_WIDTH, 3)

		view_boxes = []

		def fake_build_area_view(name, frame, region):
			view_boxes.append(region.box)
			image = FakeImage(name)
			return target_selector.AreaView(
				name=name,
				region=region,
				source_crop=image,
				valid_mask=f"mask_{name}",
				result_image=image,
				crop_offset=(0, 0),
				warp_image=FakeImage(f"{name}_warp"),
			)

		old_build_area_view = target_selector._build_area_view
		target_selector._build_area_view = fake_build_area_view
		try:
			wrong_top_single = FakeCandidate(650, 100, 170, 130, 0.95)
			selected_left = FakeCandidate(200, 100, 202, 130, 0.30)
			selected_right = FakeCandidate(430, 101, 204, 130, 0.30)
			placement = target_selector.build_placement_regions(
				object(),
				[
					wrong_top_single,
					selected_left,
					selected_right,
				],
			)
			self.assertIsNotNone(placement)
			self.assertIs(placement.selected_left_candidate, selected_left)
			self.assertIs(placement.selected_right_candidate, selected_right)
			self.assertIs(view_boxes[0], selected_left.box)
			self.assertIs(view_boxes[1], selected_right.box)
			self.assertIsNot(view_boxes[0], wrong_top_single.box)
			self.assertTrue(target_selector.placement_uses_selected_pair(placement))
			self.assertEqual(placement.region_a.area, "A")
			self.assertEqual(placement.region_b.area, "B")
			self.assertLess(placement.region_a.center[0], placement.region_b.center[0])
		finally:
			target_selector._build_area_view = old_build_area_view

	def test_area_results_are_built_from_selected_pair_images(self):
		class FakeImage:
			def __init__(self, label):
				self.label = label
				self.shape = (config.REGION_WARP_HEIGHT, config.REGION_WARP_WIDTH, 3)

			def copy(self):
				return FakeImage(f"{self.label}_result")

		class FakeDetection:
			def __init__(self, label):
				self.shape_name = "cube"
				self.color_name = "pink"
				self.center = (20 if label == "A" else 40, 30)
				self.score = 1.0

		def fake_detect_all_objects(
			image,
			valid_mask=None,
			area_name=None,
			enable_shadow_cube_fallback=False,
			task_id=None,
			expected_shape=None,
			expected_color=None,
		):
			if image.label == "selected_A":
				return [FakeDetection("A")], {}
			if image.label == "selected_B":
				return [FakeDetection("B")], {}
			return [], {}

		fake_detector = types.SimpleNamespace(
			REGION_WARP_WIDTH=config.REGION_WARP_WIDTH,
			REGION_WARP_HEIGHT=config.REGION_WARP_HEIGHT,
			detect_all_objects=fake_detect_all_objects,
			draw_detection=lambda image, detection, index: None,
		)
		old_detector = sys.modules.get("detector")
		sys.modules["detector"] = fake_detector
		try:
			area_a_image = FakeImage("selected_A")
			area_b_image = FakeImage("selected_B")
			area_a_region = target_selector.PlacementRegion(
				box=FakeCandidate(200, 100, 170, 130, 0.8).box,
				score=1.0,
				center=(200.0, 100.0),
				area="A",
			)
			area_b_region = target_selector.PlacementRegion(
				box=FakeCandidate(390, 100, 170, 130, 0.8).box,
				score=1.0,
				center=(390.0, 100.0),
				area="B",
			)
			placement = target_selector.PlacementRegions(
				region_a=area_a_region,
				region_b=area_b_region,
				area_a_image=area_a_image,
				area_b_image=area_b_image,
				pair_score=1.0,
				pair_evaluation=None,
				selected_left_candidate=None,
				selected_right_candidate=None,
				area_a_view=target_selector.AreaView(
					name="A",
					region=area_a_region,
					source_crop=area_a_image,
					valid_mask="mask_A",
					result_image=area_a_image.copy(),
					crop_offset=(100, 20),
				),
				area_b_view=target_selector.AreaView(
					name="B",
					region=area_b_region,
					source_crop=area_b_image,
					valid_mask="mask_B",
					result_image=area_b_image.copy(),
					crop_offset=(300, 20),
				),
			)
			result = target_selector.build_scene_result_from_placement(placement)
			self.assertEqual(result.area_a_result.label, "selected_A_result")
			self.assertEqual(result.area_b_result.label, "selected_B_result")
			self.assertEqual([item.area for item in result.detections], ["A", "B"])
			self.assertEqual(result.detections[0].global_center_x, 120.0)
			self.assertEqual(result.detections[1].global_center_x, 340.0)
		finally:
			if old_detector is None:
				del sys.modules["detector"]
			else:
				sys.modules["detector"] = old_detector

	def test_crop_polygon_region_masks_outside_polygon_when_cv2_available(self):
		try:
			import numpy as np
		except ModuleNotFoundError:
			self.skipTest("numpy is not installed in this Python environment")

		frame = np.full((80, 120, 3), 255, dtype=np.uint8)
		box = np.array(
			[
				[30, 20],
				[90, 20],
				[90, 60],
				[30, 60],
			],
			dtype=np.float32,
		)
		old_erode = config.VALID_REGION_ERODE_PX
		try:
			config.VALID_REGION_ERODE_PX = 0
			crop, valid_mask, offset = target_selector.crop_polygon_region(frame, box)
			self.assertEqual(offset, (30, 20))
			self.assertEqual(crop.shape[:2], valid_mask.shape[:2])
			self.assertEqual(int(valid_mask[20, 30]), 255)
			self.assertEqual(int(crop[20, 30, 0]), 255)
		finally:
			config.VALID_REGION_ERODE_PX = old_erode

	def test_area_warp_switch_does_not_change_source_crop_selection(self):
		class FakeImage:
			def __init__(self, label):
				self.label = label

			def copy(self):
				return FakeImage(f"{self.label}_copy")

		def fake_build_area_view(name, frame, region):
			image = FakeImage(f"{name}_source")
			return target_selector.AreaView(
				name=name,
				region=region,
				source_crop=image,
				valid_mask=f"mask_{name}",
				result_image=image.copy(),
				crop_offset=(0, 0),
				warp_image=FakeImage(f"{name}_warp"),
			)

		old_switch = config.SHOW_AREA_WARP_WINDOWS
		old_build_area_view = target_selector._build_area_view
		target_selector._build_area_view = fake_build_area_view
		try:
			config.SHOW_AREA_WARP_WINDOWS = False
			placement = target_selector.build_placement_regions(
				object(),
				[
					FakeCandidate(200, 100, 170, 130, 0.8),
					FakeCandidate(390, 100, 170, 130, 0.8),
				],
			)
			self.assertEqual(placement.area_a_image.label, "A_source")
			self.assertEqual(placement.area_b_image.label, "B_source")
			self.assertEqual(placement.area_a_warp.label, "A_warp")
			self.assertEqual(placement.area_b_warp.label, "B_warp")
		finally:
			config.SHOW_AREA_WARP_WINDOWS = old_switch
			target_selector._build_area_view = old_build_area_view


class DebugDisplaySwitchTests(unittest.TestCase):
	def test_area_windows_do_not_use_full_result_as_waiting_fallback(self):
		source = (ROOT / "main.py").read_text(encoding="utf-8")
		self.assertIn("np.zeros", source)
		self.assertIn("area_A waiting for A/B region", source)
		self.assertNotIn("return _make_waiting_window_image(fallback_image", source)
		self.assertIn("task_runner.area_b_image", source)
		self.assertIn("task_runner.area_b_result", source)

	def test_vision_runner_publishes_current_pair_images_before_stable(self):
		class FakeImage:
			def __init__(self, label):
				self.label = label

			def copy(self):
				return FakeImage(f"{self.label}_copy")

		region_a = target_selector.PlacementRegion(
			box=FakeCandidate(200, 100, 170, 130, 0.8).box,
			score=1.0,
			center=(200.0, 100.0),
			area="A",
		)
		region_b = target_selector.PlacementRegion(
			box=FakeCandidate(390, 100, 170, 130, 0.8).box,
			score=1.0,
			center=(390.0, 100.0),
			area="B",
		)
		area_a = FakeImage("current_A")
		area_b = FakeImage("current_B")
		placement = target_selector.PlacementRegions(
			region_a=region_a,
			region_b=region_b,
			area_a_image=area_a,
			area_b_image=area_b,
			pair_score=1.0,
			pair_evaluation=None,
			selected_left_candidate=None,
			selected_right_candidate=None,
			area_a_view=target_selector.AreaView(
				name="A",
				region=region_a,
				source_crop=area_a,
				valid_mask=None,
				result_image=area_a.copy(),
				crop_offset=(10, 20),
				crop_rect=(10, 20, 100, 80),
			),
			area_b_view=target_selector.AreaView(
				name="B",
				region=region_b,
				source_crop=area_b,
				valid_mask=None,
				result_image=area_b.copy(),
				crop_offset=(220, 20),
				crop_rect=(220, 20, 100, 80),
			),
			area_a_warp=FakeImage("warp_A"),
			area_b_warp=FakeImage("warp_B"),
		)
		runner = vision_task.VisionTaskRunner()
		runner._publish_placement_images(placement)
		self.assertIs(runner.area_a_image, area_a)
		self.assertIs(runner.area_b_image, area_b)
		self.assertEqual(runner.area_a_result.label, "current_A_copy")
		self.assertEqual(runner.area_b_result.label, "current_B_copy")
		self.assertNotEqual(runner.area_a_result.label, runner.area_b_result.label)

	def test_direct_area_preview_publishes_current_regions_without_task(self):
		class FakeImage:
			def __init__(self, label):
				self.label = label

			def copy(self):
				return FakeImage(f"{self.label}_copy")

		def fake_build_area_view(name, frame, region):
			image = FakeImage(f"current_{name}")
			return target_selector.AreaView(
				name=name,
				region=region,
				source_crop=image,
				valid_mask=f"mask_{name}",
				result_image=image.copy(),
				crop_offset=(0, 0),
				crop_rect=(0, 0, 10, 10),
				warp_image=FakeImage(f"warp_{name}"),
			)

		old_switch = config.AREA_PREVIEW_USE_STABLE_REGIONS
		old_build_area_view = target_selector._build_area_view
		target_selector._build_area_view = fake_build_area_view
		try:
			config.AREA_PREVIEW_USE_STABLE_REGIONS = False
			runner = vision_task.VisionTaskRunner()
			updated = runner.update_region_preview(
				object(),
				[
					FakeCandidate(200, 100, 170, 130, 0.8),
					FakeCandidate(390, 100, 170, 130, 0.8),
				],
			)
			self.assertTrue(updated)
			self.assertIsNone(runner.request)
			self.assertTrue(runner.has_valid_regions)
			self.assertFalse(runner.has_detected_objects)
			self.assertEqual(runner.area_a_image.label, "current_A")
			self.assertEqual(runner.area_b_image.label, "current_B")
			self.assertEqual(runner.area_a_result.label, "current_A_copy")
			self.assertEqual(runner.area_b_result.label, "current_B_copy")
		finally:
			config.AREA_PREVIEW_USE_STABLE_REGIONS = old_switch
			target_selector._build_area_view = old_build_area_view

	def test_direct_area_preview_default_uses_current_regions(self):
		self.assertFalse(config.AREA_PREVIEW_USE_STABLE_REGIONS)

	def test_stable_area_preview_mode_uses_stable_regions(self):
		class FakeImage:
			def __init__(self, label):
				self.label = label

			def copy(self):
				return FakeImage(f"{self.label}_copy")

		def fake_build_area_view(name, frame, region):
			image = FakeImage(f"{region.area}_{name}")
			return target_selector.AreaView(
				name=name,
				region=region,
				source_crop=image,
				valid_mask=None,
				result_image=image.copy(),
				crop_offset=(0, 0),
				crop_rect=(0, 0, 10, 10),
				warp_image=FakeImage(f"{region.area}_{name}_warp"),
			)

		stable_a = target_selector.PlacementRegion(
			box=FakeCandidate(200, 100, 170, 130, 0.8).box,
			score=1.0,
			center=(200.0, 100.0),
			area="stable_A",
		)
		stable_b = target_selector.PlacementRegion(
			box=FakeCandidate(390, 100, 170, 130, 0.8).box,
			score=1.0,
			center=(390.0, 100.0),
			area="stable_B",
		)
		old_switch = config.AREA_PREVIEW_USE_STABLE_REGIONS
		old_build_area_view = target_selector._build_area_view
		target_selector._build_area_view = fake_build_area_view
		try:
			config.AREA_PREVIEW_USE_STABLE_REGIONS = True
			runner = vision_task.VisionTaskRunner()
			runner.stable_regions = [stable_a, stable_b]
			updated = runner.update_region_preview(
				object(),
				[
					FakeCandidate(900, 100, 170, 130, 0.8),
					FakeCandidate(1090, 100, 170, 130, 0.8),
				],
			)
			self.assertTrue(updated)
			self.assertEqual(runner.area_a_image.label, "stable_A_A")
			self.assertEqual(runner.area_b_image.label, "stable_B_B")
		finally:
			config.AREA_PREVIEW_USE_STABLE_REGIONS = old_switch
			target_selector._build_area_view = old_build_area_view

	def test_area_preview_keeps_previous_image_when_no_valid_pair(self):
		class FakeImage:
			def __init__(self, label):
				self.label = label

		runner = vision_task.VisionTaskRunner()
		runner.area_a_image = FakeImage("old_A")
		runner.area_b_image = FakeImage("old_B")
		updated = runner.update_region_preview(object(), [])
		self.assertFalse(updated)
		self.assertEqual(runner.area_a_image.label, "old_A")
		self.assertEqual(runner.area_b_image.label, "old_B")

	def test_main_fetches_candidates_for_area_preview_windows(self):
		source = (ROOT / "main.py").read_text(encoding="utf-8")
		self.assertIn("area_preview_enabled", source)
		self.assertIn("or area_preview_enabled", source)
		self.assertIn("task_runner.update_region_preview", source)

	def test_white_single_candidate_drawing_is_hidden_in_source_window(self):
		source = (ROOT / "first.py").read_text(encoding="utf-8")
		draw_source = source.split("def draw_frame_candidates(", 1)[1].split("def warp_frame(", 1)[0]
		self.assertNotIn("candidate.score:.2f", draw_source)
		self.assertNotIn("candidate.area_ratio:.2f", draw_source)
		self.assertNotIn("candidate.rectangularity:.2f", draw_source)
		self.assertNotIn("candidate.border_score:.2f", draw_source)
		self.assertIn("(0, 255, 0)", draw_source)

	def test_main_keeps_single_candidate_tracker_logic_without_white_overlay(self):
		source = (ROOT / "main.py").read_text(encoding="utf-8")
		self.assertIn("frame_tracker = detector.FrameTracker()", source)
		self.assertIn("frame_tracker.update(candidates)", source)
		self.assertIn("detector.warp_frame(frame, frame_tracker.points)", source)
		self.assertNotIn("Frame score=", source)
		self.assertNotIn("[frame_tracker.points.astype(np.int32)]", source)

	def test_display_switches_are_independent_booleans(self):
		for name in [
			"DEBUG_DRAW_ENABLED",
			"SHOW_SOURCE_CANDIDATES_WINDOW",
			"SHOW_RESULT_WINDOW",
			"SHOW_AREA_WINDOWS",
			"SHOW_AREA_RESULT_WINDOWS",
			"SHOW_BLACK_MASK_WINDOW",
			"SHOW_COLOR_MASK_WINDOWS",
		]:
			self.assertIsInstance(getattr(config, name), bool)

	def test_draw_switches_do_not_change_pair_result(self):
		candidates = [
			FakeCandidate(200, 100, 170, 130, 0.8),
			FakeCandidate(390, 100, 170, 130, 0.8),
		]
		old_values = (
			config.DEBUG_DRAW_ENABLED,
			config.DEBUG_DRAW_ALL_CANDIDATES,
			config.DEBUG_DRAW_ALL_PAIRS,
			config.DEBUG_DRAW_PAIR_TEXT,
			config.SHOW_SOURCE_CANDIDATES_WINDOW,
		)
		try:
			config.DEBUG_DRAW_ENABLED = False
			config.DEBUG_DRAW_ALL_CANDIDATES = False
			config.DEBUG_DRAW_ALL_PAIRS = False
			config.DEBUG_DRAW_PAIR_TEXT = False
			config.SHOW_SOURCE_CANDIDATES_WINDOW = False
			hidden = target_selector.select_two_regions(candidates)

			config.DEBUG_DRAW_ENABLED = True
			config.DEBUG_DRAW_ALL_CANDIDATES = True
			config.DEBUG_DRAW_ALL_PAIRS = True
			config.DEBUG_DRAW_PAIR_TEXT = True
			config.SHOW_SOURCE_CANDIDATES_WINDOW = True
			visible = target_selector.select_two_regions(candidates)

			self.assertEqual([item.area for item in hidden], ["A", "B"])
			self.assertEqual([item.area for item in visible], ["A", "B"])
			self.assertAlmostEqual(hidden[0].center[0], visible[0].center[0])
		finally:
			(
				config.DEBUG_DRAW_ENABLED,
				config.DEBUG_DRAW_ALL_CANDIDATES,
				config.DEBUG_DRAW_ALL_PAIRS,
				config.DEBUG_DRAW_PAIR_TEXT,
				config.SHOW_SOURCE_CANDIDATES_WINDOW,
			) = old_values

	def test_window_switches_control_imshow_calls(self):
		calls = []

		class FakeCv2:
			class error(Exception):
				pass

			@staticmethod
			def imshow(name, image):
				calls.append(("imshow", name))

			@staticmethod
			def destroyWindow(name):
				calls.append(("destroy", name))

			@staticmethod
			def destroyAllWindows():
				calls.append(("destroy_all", None))

			@staticmethod
			def waitKey(delay):
				return 255

		old_cv2 = sys.modules.get("cv2")
		old_display = sys.modules.get("display")
		sys.modules["cv2"] = FakeCv2
		if "display" in sys.modules:
			del sys.modules["display"]

		try:
			display = importlib.import_module("display")
			state = display.DisplayState()
			config.SHOW_SOURCE_CANDIDATES_WINDOW = False
			config.SHOW_RESULT_WINDOW = False
			config.SHOW_COLOR_MASK_WINDOWS = False
			config.SHOW_BLACK_MASK_WINDOW = False
			display.show_images("source", "result", {}, "black", state)
			self.assertNotIn(("imshow", config.WINDOW_SOURCE), calls)
			self.assertNotIn(("imshow", config.WINDOW_RESULT), calls)

			config.SHOW_SOURCE_CANDIDATES_WINDOW = True
			display.show_images("source", "result", {}, "black", state)
			self.assertIn(("imshow", config.WINDOW_SOURCE), calls)

			calls.clear()
			display.show_window(config.WINDOW_AREA_A_RESULT, None, True)
			self.assertNotIn(("destroy", config.WINDOW_AREA_A_RESULT), calls)

			display.close_window_if_hidden("never_created", False)
		finally:
			config.SHOW_SOURCE_CANDIDATES_WINDOW = False
			config.SHOW_RESULT_WINDOW = False
			config.SHOW_COLOR_MASK_WINDOWS = False
			config.SHOW_BLACK_MASK_WINDOW = False

			if old_cv2 is None:
				del sys.modules["cv2"]
			else:
				sys.modules["cv2"] = old_cv2

			if old_display is None:
				sys.modules.pop("display", None)
			else:
				sys.modules["display"] = old_display


if __name__ == "__main__":
	unittest.main()

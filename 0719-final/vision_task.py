import statistics
import time
from dataclasses import dataclass

import config
import coordinate
import target_selector
import task_state


@dataclass
class VisionTaskStatus:
	active: bool = False
	state_text: str = "idle"
	task_id: int = 0
	target_index: int = -1
	home_count: int = 0
	target_count: int = 0
	target_desc: str = ""
	x_cm: float | None = None
	y_cm: float | None = None
	message: str = ""


class VisionTaskRunner:
	def __init__(self) -> None:
		self.request: task_state.DetectionRequest | None = None
		self.started_at = 0.0
		self.region_history: list[list] = []
		self.target_history: list[target_selector.SceneDetection] = []
		self.stable_regions = []
		self.stable_placement = None
		self.area_stabilizers = {}
		self.area_a_image = None
		self.area_b_image = None
		self.area_a_result = None
		self.area_b_result = None
		self.area_a_warp = None
		self.area_b_warp = None
		self.has_valid_regions = False
		self.has_detected_objects = False
		self.last_logged_stage = None
		self.status = VisionTaskStatus()

	def reset_round(self) -> None:
		import detector

		self.region_history.clear()
		self.target_history.clear()
		self.stable_regions = []
		self.stable_placement = None
		self.area_a_image = None
		self.area_b_image = None
		self.area_a_result = None
		self.area_b_result = None
		self.area_a_warp = None
		self.area_b_warp = None
		self.has_valid_regions = False
		self.has_detected_objects = False
		self.area_stabilizers = {
			"A": detector.ObjectStabilizer(),
			"B": detector.ObjectStabilizer(),
		}

	def cancel(self) -> None:
		self.request = None
		self.reset_round()
		self.status = VisionTaskStatus()

	def start(self, request: task_state.DetectionRequest, target_count: int) -> None:
		self.request = request
		self.started_at = time.monotonic()
		self.last_logged_stage = None
		self.reset_round()
		self.status = VisionTaskStatus(
			active=True,
			state_text="calibrating",
			task_id=request.task_id,
			target_index=request.target_index,
			home_count=request.home_count,
			target_count=target_count,
			message="waiting stable A/B inner frames",
		)
		self._log_stage_once(
			f"detect_start:{request.task_id}:{request.target_index}",
			"[STAGE] 开始视觉识别 "
			f"task=0x{request.task_id:02X}, "
			f"target={request.target_index + 1}/{target_count}",
		)

	def _log(self, message: str) -> None:
		if config.TASK_LOG_ENABLED:
			print(message)

	def _log_verbose(self, message: str) -> None:
		if config.VERBOSE_VISION_LOG:
			print(message)

	def _log_stage_once(self, stage_key: str, message: str) -> None:
		if self.last_logged_stage == stage_key:
			return

		self.last_logged_stage = stage_key
		self._log(message)

	def _publish_placement_images(self, placement: target_selector.PlacementRegions) -> None:
		self.area_a_image = placement.area_a_image
		self.area_b_image = placement.area_b_image
		self.area_a_warp = placement.area_a_warp
		self.area_b_warp = placement.area_b_warp
		self.area_a_result = (
			placement.area_a_view.source_crop.copy()
			if placement.area_a_view is not None
			else None
		)
		self.area_b_result = (
			placement.area_b_view.source_crop.copy()
			if placement.area_b_view is not None
			else None
		)
		self.has_valid_regions = (
			self.area_a_image is not None
			and self.area_b_image is not None
		)
		self.has_detected_objects = False

	def _publish_scene_result(self, scene_result: target_selector.SceneBuildResult) -> None:
		if scene_result.placement is None:
			return

		self._publish_placement_images(scene_result.placement)
		self.area_a_result = scene_result.area_a_result
		self.area_b_result = scene_result.area_b_result
		self.has_detected_objects = bool(scene_result.detections)

	def update_region_preview(self, frame, frame_candidates) -> bool:
		placement = None
		source = "current"

		if (
			config.AREA_PREVIEW_USE_STABLE_REGIONS
			and len(self.stable_regions) >= 2
		):
			source = "stable"
			placement = target_selector.build_placement_from_regions(
				frame,
				self.stable_regions,
			)
		else:
			placement = target_selector.build_placement_regions(
				frame,
				frame_candidates,
			)

		if placement is None:
			if frame_candidates:
				self._log_verbose("[CALIB] GREEN REGION EXISTS BUT CROP FAILED or pair rejected")
			return False

		self._publish_placement_images(placement)

		if config.VERBOSE_VISION_LOG:
			try:
				self._log_verbose(
					"[CALIB] preview "
					f"source={source}, "
					f"has_valid_regions={self.has_valid_regions}, "
					f"A rect={placement.area_a_view.crop_rect if placement.area_a_view else None}, "
					f"B rect={placement.area_b_view.crop_rect if placement.area_b_view else None}"
				)
			except Exception:
				pass

		return True

	def process_frame(self, frame, frame_candidates, controller: task_state.VisionTaskController) -> None:
		if self.request is None:
			self.status.active = False
			return

		if time.monotonic() - self.started_at > config.DETECTION_TIMEOUT_SEC:
			self._log_stage_once(
				f"timeout:{self.request.task_id}:{self.request.target_index}",
				"[ERROR] detection timeout, reset current visual caches and retry",
			)
			self.started_at = time.monotonic()
			self.reset_round()
			self.last_logged_stage = None
			self.status.state_text = "timeout_retry"
			self.status.message = "timeout, retrying"
			return

		placement = self._update_regions(frame, frame_candidates)

		if placement is None:
			return

		expected_shape, expected_color = target_selector.expected_detection_hint(
			self.request.task_id,
			self.request.target_index,
			list(self.request.config_queue),
		)
		scene_result = target_selector.build_scene_result_from_placement(
			placement,
			self.area_stabilizers,
			task_id=self.request.task_id,
			expected_shape=expected_shape,
			expected_color=expected_color,
		)
		self._publish_scene_result(scene_result)
		target = target_selector.select_target_for_task(
			self.request.task_id,
			self.request.target_index,
			scene_result.detections,
			list(self.request.config_queue),
			controller.completed_targets,
		)

		if target is None:
			self.target_history.clear()
			self.status.state_text = "search_target"
			self.status.message = "target not found"
			self._log_verbose("[TARGET] target not found in this frame")
			return

		stable_target = self._update_target(target)

		if stable_target is None:
			return

		self._send_stable_target(stable_target, controller)

	def _update_regions(self, frame, frame_candidates):
		placement = target_selector.build_placement_regions(frame, frame_candidates)

		if placement is None:
			evaluations = target_selector.evaluate_candidate_pairs(frame_candidates)
			reason = "no candidate pair"
			if evaluations:
				reason = evaluations[0].reject_reason or "pair rejected"
			self.region_history.clear()
			self.stable_regions = []
			self.status.state_text = "calibrating"
			self.status.message = f"need paired A/B inner frames: {reason}"
			return None

		self._publish_placement_images(placement)

		regions = [
			placement.region_a,
			placement.region_b,
		]
		self.region_history.append(regions)
		self.region_history = self.region_history[-config.CALIBRATION_STABLE_FRAMES:]

		if len(self.region_history) < config.CALIBRATION_STABLE_FRAMES:
			self.status.state_text = "calibrating"
			self.status.message = f"A/B frames stable {len(self.region_history)}/{config.CALIBRATION_STABLE_FRAMES}"
			return None

		if not self._regions_are_stable():
			self.region_history = self.region_history[-1:]
			self.stable_regions = []
			self.stable_placement = None
			self.status.state_text = "calibrating"
			self.status.message = "A/B frame jitter too large"
			return None

		self.stable_regions = self.region_history[-1]
		if self.stable_placement is None:
			self.stable_placement = placement
		self.status.state_text = "search_target"
		self.status.message = f"A/B inner frames stable pair={placement.pair_score:.2f}"
		if self.request is not None:
			self._log_stage_once(
				f"regions_stable:{self.request.task_id}:{self.request.target_index}",
				"[STAGE] A/B regions stable, searching target",
			)
			self._log_stage_once(
				f"origin:{self.request.task_id}:{self.request.target_index}",
				f"[ORIGIN] mechanical origin: AB mid-top + {config.ARM_ORIGIN_ABOVE_TOP_CM:.2f} cm upward",
			)
		return placement

	def _regions_are_stable(self) -> bool:
		import detector

		first_pair = self.region_history[0]

		for pair in self.region_history[1:]:
			for index in range(2):
				distance = detector.average_corner_distance(
					first_pair[index].box,
					pair[index].box,
				)

				if distance > config.MAX_CENTER_JITTER_PX:
					return False

		return True

	def _update_target(self, target: target_selector.SceneDetection) -> target_selector.SceneDetection | None:
		if self.target_history and not self._same_target(self.target_history[-1], target):
			self.target_history.clear()

		self.target_history.append(target)
		self.target_history = self.target_history[-config.TARGET_STABLE_FRAMES:]

		self.status.state_text = "target_stabilizing"
		self.status.target_desc = f"{target.area} {target.color} {target.shape}"
		self.status.message = f"target stable {len(self.target_history)}/{config.TARGET_STABLE_FRAMES}"

		if len(self.target_history) < config.TARGET_STABLE_FRAMES:
			return None

		center_x_values = [item.global_center_x for item in self.target_history]
		center_y_values = [item.global_center_y for item in self.target_history]

		if max(center_x_values) - min(center_x_values) > config.MAX_CENTER_JITTER_PX:
			self.target_history = self.target_history[-1:]
			self.status.message = "target x jitter too large"
			return None

		if max(center_y_values) - min(center_y_values) > config.MAX_CENTER_JITTER_PX:
			self.target_history = self.target_history[-1:]
			self.status.message = "target y jitter too large"
			return None

		latest = self.target_history[-1]
		return target_selector.SceneDetection(
			shape=latest.shape,
			color=latest.color,
			center_px=latest.center_px,
			global_center_x=float(statistics.median(center_x_values)),
			global_center_y=float(statistics.median(center_y_values)),
			area=latest.area,
			score=latest.score,
			stable_frames=len(self.target_history),
			is_shadow_cube_fallback=latest.is_shadow_cube_fallback,
			is_pink_lab_fallback=latest.is_pink_lab_fallback,
		)

	@staticmethod
	def _same_target(left: target_selector.SceneDetection, right: target_selector.SceneDetection) -> bool:
		if left.shape != right.shape or left.color != right.color or left.area != right.area:
			return False

		return (
			abs(left.global_center_x - right.global_center_x) <= config.MAX_CENTER_JITTER_PX
			and abs(left.global_center_y - right.global_center_y) <= config.MAX_CENTER_JITTER_PX
		)

	def _send_stable_target(self, target: target_selector.SceneDetection, controller: task_state.VisionTaskController) -> None:
		try:
			if self.stable_placement is None:
				raise coordinate.CalibrationMissingError("A/B planar calibration unavailable")

			coords = [
				coordinate.frame_point_to_arm_coordinate(
					item.global_center_x,
					item.global_center_y,
					self.stable_placement,
				)
				for item in self.target_history
			]
		except coordinate.CalibrationMissingError as exc:
			self.status.state_text = "calibration_missing"
			self.status.message = str(exc)
			controller.state = task_state.TaskState.ERROR
			self._log_stage_once(
				f"calibration_missing:{self.request.task_id if self.request else 0}:{self.request.target_index if self.request else -1}",
				"[CALIB] coordinate unavailable, not sending: "
				f"{exc}; target_px={target.center_px}, "
				f"scene=({target.global_center_x:.1f}, {target.global_center_y:.1f}), "
				f"target={target.area} {target.color} {target.shape}"
			)

			return

		x_values = [item.x_cm for item in coords]
		y_values = [item.y_cm for item in coords]

		if max(x_values) - min(x_values) > config.MAX_COORD_X_JITTER_CM:
			self.target_history = self.target_history[-1:]
			self.status.message = "x jitter too large"
			return

		if max(y_values) - min(y_values) > config.MAX_COORD_Y_JITTER_CM:
			self.target_history = self.target_history[-1:]
			self.status.message = "y jitter too large"
			return

		x_cm = float(statistics.median(x_values))
		y_cm = float(statistics.median(y_values))
		controller.record_completed_target(target.global_center_x, target.global_center_y)
		self.status.state_text = "sent"
		self.status.x_cm = x_cm
		self.status.y_cm = y_cm
		self.status.message = "coordinate sent"
		self._log(
			"[TARGET] "
			f"target {controller.current_target_index + 1}/{controller.target_count} stable: "
			f"{target.area} {target.color} {target.shape}, "
			f"target_px={target.center_px}, "
			f"scene=({target.global_center_x:.1f}, {target.global_center_y:.1f})"
		)
		if target.is_shadow_cube_fallback:
			self._log_stage_once(
				f"shadow_cube:{self.request.task_id if self.request else 0}:{self.request.target_index if self.request else -1}",
				f"[TARGET] shadow-resistant cube fallback used: area={target.area}",
			)
		if target.is_pink_lab_fallback:
			self._log_stage_once(
				f"pink_lab:{self.request.task_id if self.request else 0}:{self.request.target_index if self.request else -1}",
				f"[TARGET] pink LAB cube fallback used: area={target.area}",
			)
		self._log(
			f"[COORD] x={x_cm:.2f} cm, y={y_cm:.2f} cm"
		)
		controller.send_target_coordinate(x_cm, y_cm)
		self._log(
			"[STAGE] "
			f"target {controller.current_target_index + 1}/{controller.target_count} coordinate sent, "
			"waiting next home"
		)
		self.request = None
		self.status.active = False

	def draw_debug(self, image) -> None:
		if not config.DEBUG_DRAW_SELECTED_LABELS:
			return

		import cv2
		import numpy as np

		for index, region in enumerate(self.stable_regions):
			area = getattr(region, "area", "A" if index == 0 else "B")
			center_x = int(region.center[0])
			center_y = int(region.center[1])
			cv2.putText(
				image,
				area,
				(center_x - 10, center_y),
				cv2.FONT_HERSHEY_SIMPLEX,
				1.0,
				(0, 255, 255),
				2,
				cv2.LINE_AA,
			)

		if not config.DEBUG_DRAW_GLOBAL_OBJECTS or self.stable_placement is None:
			return

		try:
			frame_to_plane = coordinate.build_unified_ab_homography(self.stable_placement)
			plane_to_frame = np.linalg.inv(frame_to_plane)
			points = np.array(
				[
					[
						[0.0, 0.0],
						[config.AB_TOTAL_WIDTH_CM, 0.0],
						[config.AB_TOTAL_WIDTH_CM, config.INNER_REGION_HEIGHT_CM],
						[0.0, config.INNER_REGION_HEIGHT_CM],
						[config.INNER_REGION_WIDTH_CM, 0.0],
						[config.INNER_REGION_WIDTH_CM, config.INNER_REGION_HEIGHT_CM],
						[config.INNER_REGION_WIDTH_CM, -config.ARM_ORIGIN_ABOVE_TOP_CM],
						[config.INNER_REGION_WIDTH_CM + 5.0, -config.ARM_ORIGIN_ABOVE_TOP_CM],
						[config.INNER_REGION_WIDTH_CM, -config.ARM_ORIGIN_ABOVE_TOP_CM - 5.0],
					]
				],
				dtype=np.float32,
			)
			mapped = cv2.perspectiveTransform(points, plane_to_frame)[0]
			outline = mapped[:4].astype(np.int32)
			divider = mapped[4:6].astype(np.int32)
			origin = tuple(mapped[6].astype(np.int32))
			x_end = tuple(mapped[7].astype(np.int32))
			y_end = tuple(mapped[8].astype(np.int32))
			cv2.polylines(image, [outline], True, (0, 180, 255), 2)
			cv2.line(image, tuple(divider[0]), tuple(divider[1]), (0, 180, 255), 2)
			cv2.circle(image, origin, 5, (0, 0, 255), -1)
			cv2.arrowedLine(image, origin, x_end, (255, 0, 0), 2, tipLength=0.25)
			cv2.arrowedLine(image, origin, y_end, (0, 0, 255), 2, tipLength=0.25)
			cv2.putText(image, "O", (origin[0] + 6, origin[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
			cv2.putText(image, "+X", (x_end[0] + 6, x_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
			cv2.putText(image, "+Y", (y_end[0] + 6, y_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
		except Exception:
			return

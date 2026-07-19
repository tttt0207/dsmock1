import time
from dataclasses import dataclass
from enum import Enum

import config
import serial_protocol as protocol


class TaskState(Enum):
	SELECT_MODE = "SELECT_MODE"
	RECEIVE_CONFIG = "RECEIVE_CONFIG"
	WAIT_HOME = "WAIT_HOME"
	READY_TO_DETECT = "READY_TO_DETECT"
	WAIT_NEXT_HOME = "WAIT_NEXT_HOME"
	ERROR = "ERROR"


@dataclass
class TargetConfig:
	slot_id: int
	shape_id: int
	color_id: int


@dataclass(frozen=True)
class DetectionRequest:
	task_id: int
	target_index: int
	home_count: int
	config_queue: tuple[TargetConfig, ...]


class VisionTaskController:
	def __init__(self, sender, clock=None) -> None:
		self.sender = sender
		self.clock = clock or time.monotonic
		self.state = TaskState.SELECT_MODE
		self.current_task = 0
		self.home_count = 0
		self.target_count = 0
		self.current_target_index = -1
		self.task_config_queue: list[TargetConfig] = []
		self.completed_targets: list[tuple[float, float]] = []
		self.last_home_event_time: float | None = None
		self.vision_reset_requested = False
		self.detection_request_pending = False

	@property
	def task_id(self) -> int:
		return self.current_task

	@property
	def configs(self) -> list[TargetConfig]:
		return self.task_config_queue

	def request_vision_reset(self) -> bool:
		value = self.vision_reset_requested
		self.vision_reset_requested = False
		return value

	def consume_detection_request(self) -> DetectionRequest | None:
		if not self.detection_request_pending:
			return None

		self.detection_request_pending = False
		return DetectionRequest(
			task_id=self.current_task,
			target_index=self.current_target_index,
			home_count=self.home_count,
			config_queue=tuple(self.task_config_queue),
		)

	def _log(self, message: str) -> None:
		if config.TASK_LOG_ENABLED:
			print(message)

	def reset_to_select_mode(self, log_message: str | None = None) -> None:
		if log_message:
			self._log(log_message)

		self.state = TaskState.SELECT_MODE
		self.current_task = 0
		self.home_count = 0
		self.target_count = 0
		self.current_target_index = -1
		self.task_config_queue.clear()
		self.completed_targets.clear()
		self.last_home_event_time = None
		self.vision_reset_requested = True
		self.detection_request_pending = False

	def handle_frame(self, frame: protocol.Frame) -> None:
		if frame.raw == protocol.FORCE_RESET_FRAME:
			self.reset_to_select_mode("[RESET] 收到复位命令，返回题目选择状态")
			return

		if frame.head == protocol.HEAD_TASK_SELECT:
			self._handle_task_select(frame)
			return

		if frame.head == protocol.HEAD_CONFIG:
			self._handle_config(frame)
			return

		if frame.raw == protocol.HOME_POSITION_FRAME:
			self._handle_home_event()

	def _handle_task_select(self, frame: protocol.Frame) -> None:
		task_id = frame.data3

		self.reset_to_select_mode()

		if task_id not in {
			protocol.TASK_BASIC_1_1,
			protocol.TASK_BASIC_1_2,
			protocol.TASK_BASIC_1_3,
			protocol.TASK_ADV_2_1,
			protocol.TASK_ADV_2_2,
			protocol.TASK_ADV_2_3,
		}:
			self.state = TaskState.ERROR
			self._log(f"[ERROR][TASK] unknown task id=0x{task_id:02X}")
			return

		self.current_task = task_id
		self.home_count = 0
		self.current_target_index = -1
		self.target_count = self._initial_target_count(task_id)
		self.task_config_queue.clear()
		self.completed_targets.clear()
		self.last_home_event_time = None
		self.detection_request_pending = False
		self.vision_reset_requested = True

		if task_id in (protocol.TASK_ADV_2_2, protocol.TASK_ADV_2_3):
			self.state = TaskState.RECEIVE_CONFIG
		else:
			self.state = TaskState.READY_TO_DETECT

		task_name = protocol.TASK_ID_TO_NAME.get(task_id, f"0x{task_id:02X}")

		if self.state == TaskState.RECEIVE_CONFIG:
			self._log(
				f"[TASK] 进入{task_name}(0x{task_id:02X})，等待目标配置"
			)
		else:
			self._log(
				f"[TASK] 进入{task_name}(0x{task_id:02X})，目标数量={self.target_count}"
			)
			self._start_next_target("[STAGE] start target ")

	def _handle_config(self, frame: protocol.Frame) -> None:
		if self.state != TaskState.RECEIVE_CONFIG:
			return

		last_flag = frame.data0
		slot_id = frame.data1
		shape_id = frame.data2
		color_id = frame.data3

		if shape_id not in (0x00, 0x01, 0x02, 0x03) or color_id not in (0x00, 0x01, 0x02, 0x03, 0x04):
			self.state = TaskState.ERROR
			self._log(
				f"[ERROR][CONFIG] invalid config shape=0x{shape_id:02X}, color=0x{color_id:02X}"
			)
			return

		self.task_config_queue.append(TargetConfig(slot_id=slot_id, shape_id=shape_id, color_id=color_id))
		self._log(
			"[CONFIG] "
			f"slot={slot_id}, "
			f"shape={protocol.SHAPE_ID_TO_NAME.get(shape_id, f'0x{shape_id:02X}')}, "
			f"color={protocol.COLOR_ID_TO_NAME.get(color_id, f'0x{color_id:02X}')}, "
			f"last_flag={last_flag}"
		)

		if self.current_task == protocol.TASK_ADV_2_2:
			self.target_count = 2
			self.state = TaskState.READY_TO_DETECT
			self._log("[STAGE] 配置接收完成，共 2 个目标")
			self._start_next_target("[STAGE] start target ")
			return

		if self.current_task == protocol.TASK_ADV_2_3 and last_flag == 0x01:
			self.target_count = len(self.task_config_queue)
			self.state = TaskState.READY_TO_DETECT
			self._log(
				f"[STAGE] 配置接收完成，共 {self.target_count} 个目标"
			)
			self._start_next_target("[STAGE] start target ")

	def _handle_home_event(self) -> None:
		if self.current_task == 0 or self.state not in (TaskState.WAIT_HOME, TaskState.WAIT_NEXT_HOME):
			return

		now = self.clock()

		if self.last_home_event_time is not None and now - self.last_home_event_time < config.HOME_EVENT_DEBOUNCE_SEC:
			if config.VERBOSE_VISION_LOG:
				self._log("[HOME] duplicate home frame ignored")
			return

		self.last_home_event_time = now
		self._start_next_target("[HOME] 收到有效回零帧，")

	def _start_next_target(self, log_prefix: str) -> None:
		self.home_count += 1

		if self.target_count <= 0 or self.home_count > self.target_count:
			task_name = protocol.TASK_ID_TO_NAME.get(
				self.current_task,
				f"0x{self.current_task:02X}",
			)
			self.reset_to_select_mode(
				f"[TASK] {task_name}(0x{self.current_task:02X}) 全部目标处理完成"
			)
			return

		self.current_target_index = self.home_count - 1
		self.vision_reset_requested = True
		self.detection_request_pending = True
		self.state = TaskState.READY_TO_DETECT
		self.last_home_event_time = self.clock()
		if log_prefix.startswith("[STAGE] start target"):
			self._log(f"{log_prefix}{self.current_target_index + 1}/{self.target_count}")
		else:
			self._log(
				f"{log_prefix}开始识别第 {self.current_target_index + 1}/{self.target_count} 个目标"
			)

	def mark_detection_started(self) -> None:
		if self.state == TaskState.READY_TO_DETECT:
			self.state = TaskState.WAIT_NEXT_HOME

	def send_target_coordinate(self, x_cm: float, y_cm: float) -> bytes:
		frame = protocol.build_coord_frame_from_xy(x_cm, y_cm)
		self.sender(frame)
		self.state = TaskState.WAIT_NEXT_HOME
		return frame

	def record_completed_target(self, scene_x: float, scene_y: float) -> None:
		self.completed_targets.append((float(scene_x), float(scene_y)))

	def poll(self) -> None:
		return

	@staticmethod
	def _initial_target_count(task_id: int) -> int:
		if task_id in (protocol.TASK_BASIC_1_1, protocol.TASK_BASIC_1_2):
			return 1

		if task_id == protocol.TASK_BASIC_1_3:
			return 4

		if task_id == protocol.TASK_ADV_2_1:
			return 3

		return 0

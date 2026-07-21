from dataclasses import dataclass


FRAME_SIZE = 6

HEAD_TASK_SELECT = 0x01
TAIL_TASK_SELECT = 0x10
HEAD_CONFIG = 0x02
TAIL_CONFIG = 0x20
HEAD_COORD = 0x03
TAIL_COORD = 0x30
HEAD_ANGLE = 0x04
TAIL_ANGLE = 0x40
HEAD_ARM = 0x05
TAIL_ARM = 0x50

VALID_RX_TAILS = {
	HEAD_TASK_SELECT: TAIL_TASK_SELECT,
	HEAD_CONFIG: TAIL_CONFIG,
	HEAD_ARM: TAIL_ARM,
}

TASK_BASIC_1_1 = 0x11
TASK_BASIC_1_2 = 0x22
TASK_BASIC_1_3 = 0x33
TASK_ADV_2_1 = 0x44
TASK_ADV_2_2 = 0x55
TASK_ADV_2_3 = 0x66

TASK_ID_TO_NAME = {
	TASK_BASIC_1_1: "基础题1-1",
	TASK_BASIC_1_2: "基础题1-2",
	TASK_BASIC_1_3: "基础题1-3",
	TASK_ADV_2_1: "发挥题2-1",
	TASK_ADV_2_2: "发挥题2-2",
	TASK_ADV_2_3: "发挥题2-3",
}

CMD_ARM_HOME_REQUEST = 0x01
CMD_FORCE_RESET = 0xFF

SHAPE_ANY = 0x00
SHAPE_CUBE = 0x05
SHAPE_CUBOID = 0x06
SHAPE_BALL = 0x07

COLOR_ANY = 0x00
COLOR_RED = 0x41
COLOR_BLUE = 0x42
COLOR_BLACK = 0x43
COLOR_YELLOW = 0x44

HOME_POSITION_FRAME = bytes([HEAD_ARM, CMD_ARM_HOME_REQUEST, 0x00, 0x00, 0x00, TAIL_ARM])
FORCE_RESET_FRAME = bytes([HEAD_ARM, CMD_FORCE_RESET, 0x00, 0x00, 0x00, TAIL_ARM])

SHAPE_ID_TO_NAME = {
	SHAPE_ANY: "any",
	SHAPE_CUBE: "cube",
	SHAPE_CUBOID: "cuboid",
	SHAPE_BALL: "ball",
}

COLOR_ID_TO_NAME = {
	COLOR_ANY: "any",
	COLOR_RED: "pink",
	COLOR_BLUE: "blue",
	COLOR_BLACK: "green",
	COLOR_YELLOW: "orange",
}

COLOR_PINK = COLOR_RED
COLOR_GREEN = COLOR_BLACK
COLOR_ORANGE = COLOR_YELLOW


@dataclass(frozen=True)
class Frame:
	head: int
	data0: int
	data1: int
	data2: int
	data3: int
	tail: int

	@property
	def raw(self) -> bytes:
		return bytes([
			self.head,
			self.data0,
			self.data1,
			self.data2,
			self.data3,
			self.tail,
		])


def bytes_to_hex(data: bytes | bytearray) -> str:
	return " ".join(f"{item:02X}" for item in data)


def parse_frame(raw: bytes | bytearray) -> Frame:
	if len(raw) != FRAME_SIZE:
		raise ValueError("frame must be 6 bytes")

	return Frame(
		head=raw[0],
		data0=raw[1],
		data1=raw[2],
		data2=raw[3],
		data3=raw[4],
		tail=raw[5],
	)


def build_coord_frame(x_raw: int, y_raw: int) -> bytes:
	if not -32768 <= x_raw <= 32767:
		raise ValueError(f"x out of int16 range: {x_raw}")

	if not -32768 <= y_raw <= 32767:
		raise ValueError(f"y out of int16 range: {y_raw}")

	x_encoded = x_raw & 0xFFFF
	y_encoded = y_raw & 0xFFFF
	return bytes([
		HEAD_COORD,
		(x_encoded >> 8) & 0xFF,
		x_encoded & 0xFF,
		(y_encoded >> 8) & 0xFF,
		y_encoded & 0xFF,
		TAIL_COORD,
	])


def build_coord_frame_from_xy(x_cm: float, y_cm: float) -> bytes:
	x_raw = int(round(float(x_cm) * 100.0))
	y_raw = int(round(float(y_cm) * 100.0))
	return build_coord_frame(x_raw, y_raw)


def build_angle_frame(angle_raw: int) -> bytes:
	if not -32768 <= angle_raw <= 32767:
		raise ValueError(f"angle out of int16 range: {angle_raw}")

	angle_encoded = angle_raw & 0xFFFF
	return bytes([
		HEAD_ANGLE,
		(angle_encoded >> 8) & 0xFF,
		angle_encoded & 0xFF,
		0x00,
		0x00,
		TAIL_ANGLE,
	])


def build_angle_frame_from_deg(angle_deg: float) -> bytes:
	angle_raw = int(round(float(angle_deg) * 100.0))
	return build_angle_frame(angle_raw)


class FrameParser:
	def __init__(self) -> None:
		self.buffer = bytearray()

	def feed(self, data: bytes | bytearray) -> list[Frame]:
		frames = []
		self.buffer.extend(data)

		while len(self.buffer) >= FRAME_SIZE:
			raw_frame = bytes(self.buffer[:FRAME_SIZE])

			if raw_frame == FORCE_RESET_FRAME:
				frames.append(parse_frame(raw_frame))
				del self.buffer[:FRAME_SIZE]
				continue

			head = self.buffer[0]
			expected_tail = VALID_RX_TAILS.get(head)

			if expected_tail is None:
				del self.buffer[0]
				continue

			if raw_frame[FRAME_SIZE - 1] != expected_tail:
				del self.buffer[0]
				continue

			if head == HEAD_ARM and raw_frame not in (HOME_POSITION_FRAME, FORCE_RESET_FRAME):
				del self.buffer[0]
				continue

			frames.append(parse_frame(raw_frame))
			del self.buffer[:FRAME_SIZE]

		return frames

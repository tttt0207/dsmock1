import queue
import threading
import time

import config
from serial_protocol import Frame, FrameParser, bytes_to_hex


try:
	import serial
except ImportError:
	serial = None


def open_serial_port():
	if serial is None:
		raise RuntimeError("pyserial is not installed")

	return serial.Serial(
		port=config.SERIAL_PORT,
		baudrate=config.SERIAL_BAUD,
		timeout=config.SERIAL_TIMEOUT,
	)


class SerialManager:
	def __init__(self) -> None:
		self.serial_port = None
		self.stop_event = threading.Event()
		self.rx_queue: queue.Queue[Frame] = queue.Queue()
		self.parser = FrameParser()
		self.thread: threading.Thread | None = None
		self.enabled = False
		self.last_read_error_text = ""
		self.last_read_error_time = 0.0

	def start(self) -> None:
		try:
			self.serial_port = open_serial_port()
		except Exception as exc:
			if config.SERIAL_LOG_ENABLED:
				print(f"[ERROR][SERIAL] open failed: {exc}")
			self.enabled = False
			return

		self.enabled = True
		self.stop_event.clear()
		self.thread = threading.Thread(
			target=self._receive_loop,
			name="serial-rx",
			daemon=True,
		)
		self.thread.start()
		if config.SERIAL_LOG_ENABLED:
			print(f"[SERIAL] opened port={config.SERIAL_PORT} baud={config.SERIAL_BAUD}")

	def _receive_loop(self) -> None:
		while not self.stop_event.is_set():
			try:
				waiting = self.serial_port.in_waiting if self.serial_port is not None else 0
				data = self.serial_port.read(waiting or 1) if self.serial_port is not None else b""
			except Exception as exc:
				self._log_read_error(exc)
				time.sleep(0.02)
				continue

			if not data:
				time.sleep(0.005)
				continue

			for frame in self.parser.feed(data):
				if config.SERIAL_LOG_ENABLED:
					print(f"[RX] {bytes_to_hex(frame.raw)}")
				self.rx_queue.put(frame)

	def _log_read_error(self, exc: Exception) -> None:
		if not config.SERIAL_LOG_ENABLED:
			return

		now = time.monotonic()
		text = str(exc)

		if (
			text == self.last_read_error_text
			and now - self.last_read_error_time < config.SERIAL_ERROR_LOG_INTERVAL_SEC
		):
			return

		self.last_read_error_text = text
		self.last_read_error_time = now
		print(f"[ERROR][SERIAL] read failed: {exc}")

	def get_frames(self) -> list[Frame]:
		frames = []

		while True:
			try:
				frames.append(self.rx_queue.get_nowait())
			except queue.Empty:
				break

		return frames

	def send(self, frame: bytes) -> None:
		if not self.enabled or self.serial_port is None:
			if config.SERIAL_LOG_ENABLED:
				print(f"[TX-SKIP] serial unavailable: {bytes_to_hex(frame)}")
			return

		try:
			self.serial_port.write(frame)
			if config.SERIAL_LOG_ENABLED:
				print(f"[TX] {bytes_to_hex(frame)}")
		except Exception as exc:
			if config.SERIAL_LOG_ENABLED:
				print(f"[ERROR][SERIAL] write failed: {exc}; frame={bytes_to_hex(frame)}")

	def close(self) -> None:
		self.stop_event.set()

		if self.thread is not None:
			self.thread.join(timeout=0.5)

		if self.serial_port is not None:
			try:
				self.serial_port.close()
			except Exception:
				pass

			if config.SERIAL_LOG_ENABLED:
				print("[SERIAL] closed")

import platform

import cv2

import config


def _camera_backend() -> int | None:
	if config.CAMERA_BACKEND == "v4l2":
		return cv2.CAP_V4L2

	if config.CAMERA_BACKEND == "dshow":
		return cv2.CAP_DSHOW

	if platform.system().lower() == "linux":
		return cv2.CAP_V4L2

	if platform.system().lower() == "windows":
		return cv2.CAP_DSHOW

	return None


def open_camera() -> cv2.VideoCapture:
	backend = _camera_backend()

	if backend is None:
		camera = cv2.VideoCapture(config.CAMERA_ID)
	else:
		camera = cv2.VideoCapture(config.CAMERA_ID, backend)

	if not camera.isOpened():
		raise RuntimeError(
			f"Cannot open camera {config.CAMERA_ID}. "
			"Check the device id, USB connection, and camera permissions."
		)

	camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
	camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

	actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
	actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
	print(f"Camera opened: id={config.CAMERA_ID}, size={actual_width}x{actual_height}")

	return camera


def read_frame(camera: cv2.VideoCapture) -> tuple[bool, object]:
	success, frame = camera.read()

	if not success or frame is None:
		return False, None

	return True, frame


def release_camera(camera: cv2.VideoCapture | None) -> None:
	if camera is not None:
		camera.release()

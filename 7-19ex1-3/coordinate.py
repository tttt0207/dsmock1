from __future__ import annotations

from dataclasses import dataclass

import config


@dataclass(frozen=True)
class CoordinateResult:
	x_cm: float
	y_cm: float
	x_raw: int
	y_raw: int


class CalibrationMissingError(RuntimeError):
	pass


def _order_points(points):
	import numpy as np

	points = np.asarray(points, dtype=np.float32)
	ordered = np.zeros((4, 2), dtype=np.float32)
	point_sum = points.sum(axis=1)
	point_diff = np.diff(points, axis=1).reshape(-1)
	ordered[0] = points[int(np.argmin(point_sum))]
	ordered[2] = points[int(np.argmax(point_sum))]
	ordered[1] = points[int(np.argmin(point_diff))]
	ordered[3] = points[int(np.argmax(point_diff))]
	return ordered


def _check_int16(raw_value: int, name: str) -> int:
	if not -32768 <= raw_value <= 32767:
		raise ValueError(f"{name} out of int16 range: {raw_value}")

	return raw_value


def _physical_points():
	import numpy as np

	return np.array(
		[
			[0.0, 0.0],
			[config.AB_TOTAL_WIDTH_CM, 0.0],
			[config.AB_TOTAL_WIDTH_CM, config.INNER_REGION_HEIGHT_CM],
			[0.0, config.INNER_REGION_HEIGHT_CM],
		],
		dtype=np.float32,
	)


def build_unified_ab_homography(placement):
	import cv2
	import numpy as np

	if placement is None or placement.region_a is None or placement.region_b is None:
		raise CalibrationMissingError("A/B planar calibration unavailable: missing placement")

	area_a = placement.region_a
	area_b = placement.region_b

	if float(area_a.center[0]) >= float(area_b.center[0]):
		raise CalibrationMissingError("A/B planar calibration unavailable: A is not left of B")

	box_a = _order_points(area_a.box)
	box_b = _order_points(area_b.box)
	frame_points = np.array(
		[
			box_a[0],
			box_b[1],
			box_b[2],
			box_a[3],
		],
		dtype=np.float32,
	)

	if abs(float(cv2.contourArea(frame_points))) < 1.0:
		raise CalibrationMissingError("A/B planar calibration unavailable: zero area")

	matrix = cv2.getPerspectiveTransform(
		frame_points,
		_physical_points(),
	)
	return matrix


def frame_point_to_ab_plane(frame_x: float, frame_y: float, placement) -> tuple[float, float]:
	import cv2
	import math
	import numpy as np

	matrix = build_unified_ab_homography(placement)
	point = np.array([[[float(frame_x), float(frame_y)]]], dtype=np.float32)
	plane_point = cv2.perspectiveTransform(point, matrix)[0][0]
	u_cm = float(plane_point[0])
	v_cm = float(plane_point[1])

	if not (math.isfinite(u_cm) and math.isfinite(v_cm)):
		raise CalibrationMissingError("A/B planar calibration unavailable: non-finite result")

	return u_cm, v_cm


def plane_point_to_arm_xy(u_cm: float, v_cm: float) -> tuple[float, float]:
	x_cm = float(u_cm) - config.INNER_REGION_WIDTH_CM
	y_cm = -config.ARM_ORIGIN_ABOVE_TOP_CM - float(v_cm)
	return x_cm, y_cm


def frame_point_to_arm_coordinate(frame_x: float, frame_y: float, placement) -> CoordinateResult:
	u_cm, v_cm = frame_point_to_ab_plane(frame_x, frame_y, placement)
	x_cm, y_cm = plane_point_to_arm_xy(u_cm, v_cm)
	x_raw = _check_int16(int(round(x_cm * 100.0)), "x")
	y_raw = _check_int16(int(round(y_cm * 100.0)), "y")
	return CoordinateResult(
		x_cm=x_cm,
		y_cm=y_cm,
		x_raw=x_raw,
		y_raw=y_raw,
	)

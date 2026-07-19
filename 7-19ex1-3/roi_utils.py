import config


def clamp_ratio(value: float) -> float:
	return max(0.0, min(1.0, float(value)))


def get_placement_roi_bounds(
	image_width: int,
	image_height: int,
) -> tuple[int, int, int, int]:
	x_min = clamp_ratio(config.PLACEMENT_ROI_X_MIN_RATIO)
	x_max = clamp_ratio(config.PLACEMENT_ROI_X_MAX_RATIO)
	y_min = clamp_ratio(config.PLACEMENT_ROI_Y_MIN_RATIO)
	y_max = clamp_ratio(config.PLACEMENT_ROI_Y_MAX_RATIO)

	if x_min >= x_max:
		x_min, x_max = 0.0, 1.0

	if y_min >= y_max:
		y_min, y_max = 0.0, 1.0

	x0 = int(round(image_width * x_min))
	x1 = int(round(image_width * x_max))
	y0 = int(round(image_height * y_min))
	y1 = int(round(image_height * y_max))

	x0 = max(0, min(image_width - 1, x0))
	x1 = max(x0 + 1, min(image_width, x1))
	y0 = max(0, min(image_height - 1, y0))
	y1 = max(y0 + 1, min(image_height, y1))

	return x0, y0, x1, y1


def offset_point_to_global(
	local_x: float,
	local_y: float,
	roi_x0: int,
	roi_y0: int,
) -> tuple[float, float]:
	return local_x + roi_x0, local_y + roi_y0

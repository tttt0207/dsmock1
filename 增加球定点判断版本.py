import math
from collections import Counter, deque
from dataclasses import dataclass

import cv2
import numpy as np


# ============================================================
# 只需要优先修改这里
# ============================================================

CAMERA_ID = 1

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# 黑框实际尺寸
FRAME_WIDTH_CM = 20.0
FRAME_HEIGHT_CM = 10.0
BLACK_BORDER_CM = 1.4

# 透视矫正后的尺寸
FRAME_WARP_WIDTH = 800
FRAME_WARP_HEIGHT = 400

# 黑色阈值：反光较强时可将 BLACK_V_MAX 从 95 调到 110~125
BLACK_H_MIN = 0
BLACK_S_MIN = 0
BLACK_V_MIN = 0
BLACK_H_MAX = 179
BLACK_S_MAX = 255
BLACK_V_MAX = 105

# 黑框候选限制
FRAME_RATIO_MIN = 1.45
FRAME_RATIO_MAX = 2.65
FRAME_MIN_AREA_RATIO = 0.025
FRAME_MAX_AREA_RATIO = 0.55
FRAME_MIN_SCORE = 0.38
MAX_FRAME_JUMP_PX = 90.0
REACQUIRE_MIN_SCORE = 0.62
REACQUIRE_MAX_BORDER_ERROR_CM = 0.45
REACQUIRE_RATIO_ERROR = 0.35
FRAME_REACQUIRE_CONFIRMATIONS = 3
FRAME_PENDING_MATCH_PX = 45.0
FRAME_SMOOTH_ALPHA = 0.20

# 每隔多少帧尝试更新一次黑框
FRAME_UPDATE_INTERVAL = 15

# 是否显示所有黑框候选
SHOW_FRAME_CANDIDATES = True
MAX_CANDIDATES_TO_DRAW = 8

# 彩色物体最小轮廓面积
MIN_OBJECT_AREA = 1200

# 形状阈值
BALL_CIRCULARITY_THRESHOLD = 0.68
BALL_MAX_RATIO = 1.25
BALL_MIN_VERTEX_COUNT = 8
CUBE_MAX_RATIO = 1.27
CUBOID_MIN_RATIO = 1.33
APPROX_EPSILON_RATIO = 0.02

# 多帧稳定参数
STABLE_WINDOW = 7
STABLE_MIN_COUNT = 5
TRACK_MAX_DISTANCE_PX = 150.0
TRACK_MAX_MISSED_FRAMES = 15

# 形态学核大小
COLOR_MORPH_KERNEL_SIZE = 5
BLACK_MORPH_KERNEL_SIZE = 9

# OpenCV HSV：
# H: 0~179，S: 0~255，V: 0~255
COLOR_RANGES = {
	"green": [
		((35, 60, 45), (90, 255, 255)),
	],
	"blue": [
		((90, 70, 40), (135, 255, 255)),
	],
	"yellow": [
		((18, 70, 70), (38, 255, 255)),
	],
	"pink": [
		((140, 35, 70), (179, 255, 255)),
		((0, 35, 90), (8, 255, 255)),
	],
}

COLOR_TEXT = {
	"green": "Green",
	"blue": "Blue",
	"yellow": "Yellow",
	"pink": "Pink",
}

SHAPE_TEXT = {
	"ball": "Ball",
	"cube": "Cube",
	"cuboid": "Cuboid",
	"unknown": "Unknown",
	"uncertain": "Uncertain",
}


@dataclass
class FrameCandidate:
	box: np.ndarray
	score: float
	border_cm: float
	border_error_cm: float
	ratio: float
	ring_score: float
	area_ratio: float
	center: tuple[float, float]


@dataclass
class Detection:
	color_name: str
	shape_name: str
	contour: np.ndarray
	center: tuple[int, int]
	area: float
	circularity: float
	aspect_ratio: float
	vertex_count: int
	score: float


def order_points(points: np.ndarray) -> np.ndarray:
	points = np.asarray(points, dtype=np.float32)
	ordered = np.zeros((4, 2), dtype=np.float32)

	point_sum = points.sum(axis=1)
	point_diff = np.diff(points, axis=1).reshape(-1)

	ordered[0] = points[np.argmin(point_sum)]		# 左上
	ordered[2] = points[np.argmax(point_sum)]		# 右下
	ordered[1] = points[np.argmin(point_diff)]		# 右上
	ordered[3] = points[np.argmax(point_diff)]		# 左下

	return ordered


def polygon_center(points: np.ndarray) -> tuple[float, float]:
	return (
		float(np.mean(points[:, 0])),
		float(np.mean(points[:, 1])),
	)


def average_corner_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
	a = order_points(points_a)
	b = order_points(points_b)
	return float(np.mean(np.linalg.norm(a - b, axis=1)))


def create_black_mask(frame: np.ndarray) -> np.ndarray:
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	mask = cv2.inRange(
		hsv,
		np.array(
			(BLACK_H_MIN, BLACK_S_MIN, BLACK_V_MIN),
			dtype=np.uint8,
		),
		np.array(
			(BLACK_H_MAX, BLACK_S_MAX, BLACK_V_MAX),
			dtype=np.uint8,
		),
	)

	kernel = cv2.getStructuringElement(
		cv2.MORPH_RECT,
		(BLACK_MORPH_KERNEL_SIZE, BLACK_MORPH_KERNEL_SIZE),
	)

	# 闭运算连接胶带上的小反光断口。
	mask = cv2.morphologyEx(
		mask,
		cv2.MORPH_CLOSE,
		kernel,
		iterations=2,
	)

	# 小幅开运算去除孤立噪点。
	small_kernel = cv2.getStructuringElement(
		cv2.MORPH_RECT,
		(3, 3),
	)
	mask = cv2.morphologyEx(
		mask,
		cv2.MORPH_OPEN,
		small_kernel,
		iterations=1,
	)

	return mask


def warp_binary_to_candidate(
	mask: np.ndarray,
	box: np.ndarray,
	width: int = 400,
	height: int = 200,
) -> np.ndarray:
	ordered = order_points(box)

	target = np.array(
		[
			[0, 0],
			[width - 1, 0],
			[width - 1, height - 1],
			[0, height - 1],
		],
		dtype=np.float32,
	)

	matrix = cv2.getPerspectiveTransform(ordered, target)

	return cv2.warpPerspective(
		mask,
		matrix,
		(width, height),
		flags=cv2.INTER_NEAREST,
	)


def make_ideal_ring(width: int, height: int, thickness: int) -> np.ndarray:
	ring = np.zeros((height, width), dtype=np.uint8)

	cv2.rectangle(
		ring,
		(0, 0),
		(width - 1, height - 1),
		255,
		-1,
	)

	inner_left = thickness
	inner_top = thickness
	inner_right = width - 1 - thickness
	inner_bottom = height - 1 - thickness

	if inner_right > inner_left and inner_bottom > inner_top:
		cv2.rectangle(
			ring,
			(inner_left, inner_top),
			(inner_right, inner_bottom),
			0,
			-1,
		)

	return ring


def mask_f1_score(actual: np.ndarray, ideal: np.ndarray) -> float:
	actual_bool = actual > 0
	ideal_bool = ideal > 0

	true_positive = np.count_nonzero(actual_bool & ideal_bool)
	false_positive = np.count_nonzero(actual_bool & ~ideal_bool)
	false_negative = np.count_nonzero(~actual_bool & ideal_bool)

	precision_denominator = true_positive + false_positive
	recall_denominator = true_positive + false_negative

	if precision_denominator == 0 or recall_denominator == 0:
		return 0.0

	precision = true_positive / precision_denominator
	recall = true_positive / recall_denominator

	if precision + recall == 0:
		return 0.0

	return 2.0 * precision * recall / (precision + recall)


def estimate_border_width(
	warped_mask: np.ndarray,
) -> tuple[float, float]:
	"""
	把候选框统一拉伸到 400×200。

	由于真实外框为 20×10 cm，因此两个方向都是 20 px/cm。
	1.4 cm 理论上约等于 28 px。

	遍历不同厚度的理想矩形环，选择与实际黑色掩膜最接近的厚度。
	"""
	height, width = warped_mask.shape[:2]

	pixels_per_cm_x = width / FRAME_WIDTH_CM
	pixels_per_cm_y = height / FRAME_HEIGHT_CM
	pixels_per_cm = (pixels_per_cm_x + pixels_per_cm_y) / 2.0

	min_thickness = max(4, int(0.45 * pixels_per_cm))
	max_thickness = min(
		int(min(width, height) * 0.34),
		int(2.7 * pixels_per_cm),
	)

	best_thickness = 0
	best_score = 0.0

	for thickness in range(min_thickness, max_thickness + 1):
		ideal_ring = make_ideal_ring(width, height, thickness)
		score = mask_f1_score(warped_mask, ideal_ring)

		if score > best_score:
			best_score = score
			best_thickness = thickness

	border_cm = best_thickness / pixels_per_cm if pixels_per_cm > 0 else 0.0
	return border_cm, best_score


def candidate_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
	rect_a = cv2.minAreaRect(box_a.astype(np.float32))
	rect_b = cv2.minAreaRect(box_b.astype(np.float32))

	intersection_type, intersection = cv2.rotatedRectangleIntersection(
		rect_a,
		rect_b,
	)

	if intersection_type == cv2.INTERSECT_NONE or intersection is None:
		return 0.0

	intersection_area = abs(cv2.contourArea(intersection))
	area_a = abs(cv2.contourArea(box_a.astype(np.float32)))
	area_b = abs(cv2.contourArea(box_b.astype(np.float32)))

	union = area_a + area_b - intersection_area

	if union <= 0:
		return 0.0

	return intersection_area / union


def remove_duplicate_frame_candidates(
	candidates: list[FrameCandidate],
) -> list[FrameCandidate]:
	result = []

	for candidate in candidates:
		duplicate = False

		for accepted in result:
			if candidate_iou(candidate.box, accepted.box) > 0.72:
				duplicate = True
				break

		if not duplicate:
			result.append(candidate)

	return result


def find_black_frame_candidates(
	frame: np.ndarray,
	last_frame_points: np.ndarray | None = None,
) -> tuple[list[FrameCandidate], np.ndarray]:
	black_mask = create_black_mask(frame)

	contours, _ = cv2.findContours(
		black_mask,
		cv2.RETR_LIST,
		cv2.CHAIN_APPROX_SIMPLE,
	)

	frame_area = frame.shape[0] * frame.shape[1]
	candidates = []

	for contour in contours:
		contour_area = cv2.contourArea(contour)

		if contour_area < frame_area * 0.008:
			continue

		rect = cv2.minAreaRect(contour)
		(center_x, center_y), (width, height), _ = rect

		if width < 20 or height < 20:
			continue

		long_side = max(width, height)
		short_side = min(width, height)
		ratio = long_side / short_side

		if not FRAME_RATIO_MIN <= ratio <= FRAME_RATIO_MAX:
			continue

		box = order_points(cv2.boxPoints(rect))
		box_area = abs(cv2.contourArea(box))
		area_ratio = box_area / frame_area

		if not FRAME_MIN_AREA_RATIO <= area_ratio <= FRAME_MAX_AREA_RATIO:
			continue

		warped_mask = warp_binary_to_candidate(black_mask, box)
		border_cm, ring_score = estimate_border_width(warped_mask)
		border_error_cm = abs(border_cm - BLACK_BORDER_CM)

		# 长宽比越接近 2 越好。
		ratio_score = max(
			0.0,
			1.0 - abs(ratio - 2.0) / 0.75,
		)

		# 黑边宽度越接近 1.4 cm 越好。
		border_score = math.exp(
			-((border_error_cm / 0.48) ** 2)
		)

		# 黑框通常占有一定画面面积，过小候选降权。
		size_score = min(
			1.0,
			area_ratio / 0.10,
		)

		# 综合分只评价候选本身，不把“靠近上一帧”写入候选排名。
		# 否则一旦第一帧选错，错误框会因位置优势一直锁死。
		score = (
			0.30 * ratio_score
			+ 0.36 * border_score
			+ 0.29 * ring_score
			+ 0.05 * size_score
		)

		candidates.append(
			FrameCandidate(
				box=box,
				score=score,
				border_cm=border_cm,
				border_error_cm=border_error_cm,
				ratio=ratio,
				ring_score=ring_score,
				area_ratio=area_ratio,
				center=(center_x, center_y),
			)
		)

	candidates.sort(
		key=lambda item: item.score,
		reverse=True,
	)
	candidates = remove_duplicate_frame_candidates(candidates)

	return candidates, black_mask


class FrameTracker:
	def __init__(self) -> None:
		self.points: np.ndarray | None = None
		self.candidate: FrameCandidate | None = None
		self.pending_candidate: FrameCandidate | None = None
		self.pending_count = 0

	@staticmethod
	def is_reliable(candidate: FrameCandidate) -> bool:
		return (
			candidate.score >= REACQUIRE_MIN_SCORE
			and candidate.border_error_cm
			<= REACQUIRE_MAX_BORDER_ERROR_CM
			and abs(candidate.ratio - 2.0)
			<= REACQUIRE_RATIO_ERROR
		)

	def reset(self) -> None:
		self.points = None
		self.candidate = None
		self.pending_candidate = None
		self.pending_count = 0

	def force_acquire(
		self,
		candidates: list[FrameCandidate],
	) -> FrameCandidate | None:
		valid = [
			item
			for item in candidates
			if item.score >= FRAME_MIN_SCORE
		]

		if not valid:
			return None

		selected = valid[0]
		self.points = selected.box.copy()
		self.candidate = selected
		self.pending_candidate = None
		self.pending_count = 0
		return selected

	def update(
		self,
		candidates: list[FrameCandidate],
	) -> tuple[FrameCandidate | None, bool]:
		"""
		返回：(当前黑框候选，是否发生了远距离重新捕获)。

		旧框附近的小变化会做平滑更新；
		远处的新第一名必须连续出现多次，才切换过去。
		"""
		valid = [
			item
			for item in candidates
			if item.score >= FRAME_MIN_SCORE
		]

		if not valid:
			return self.candidate, False

		best = valid[0]

		if self.points is None:
			self.points = best.box.copy()
			self.candidate = best
			return best, True

		jump = average_corner_distance(
			best.box,
			self.points,
		)

		if jump <= MAX_FRAME_JUMP_PX:
			# 同一位置的小幅抖动，用指数平滑降低透视画面跳动。
			ordered_new = order_points(best.box)
			ordered_old = order_points(self.points)

			self.points = (
				(1.0 - FRAME_SMOOTH_ALPHA) * ordered_old
				+ FRAME_SMOOTH_ALPHA * ordered_new
			).astype(np.float32)

			self.candidate = FrameCandidate(
				box=self.points.copy(),
				score=best.score,
				border_cm=best.border_cm,
				border_error_cm=best.border_error_cm,
				ratio=best.ratio,
				ring_score=best.ring_score,
				area_ratio=best.area_ratio,
				center=polygon_center(self.points),
			)

			self.pending_candidate = None
			self.pending_count = 0
			return self.candidate, False

		# 跳得很远时，只有可靠候选才能进入重新捕获流程。
		if not self.is_reliable(best):
			self.pending_candidate = None
			self.pending_count = 0
			return self.candidate, False

		if self.pending_candidate is None:
			self.pending_candidate = best
			self.pending_count = 1
			return self.candidate, False

		pending_distance = average_corner_distance(
			best.box,
			self.pending_candidate.box,
		)

		if pending_distance <= FRAME_PENDING_MATCH_PX:
			self.pending_candidate = best
			self.pending_count += 1
		else:
			self.pending_candidate = best
			self.pending_count = 1

		if self.pending_count < FRAME_REACQUIRE_CONFIRMATIONS:
			return self.candidate, False

		self.points = best.box.copy()
		self.candidate = best
		self.pending_candidate = None
		self.pending_count = 0
		return best, True

def draw_frame_candidates(
	image: np.ndarray,
	candidates: list[FrameCandidate],
	selected_box: np.ndarray | None,
) -> None:
	if not SHOW_FRAME_CANDIDATES:
		return

	for index, candidate in enumerate(
		candidates[:MAX_CANDIDATES_TO_DRAW],
		start=1,
	):
		box_int = candidate.box.astype(np.int32)

		is_selected = False

		if selected_box is not None:
			is_selected = (
				average_corner_distance(
					candidate.box,
					selected_box,
				)
				< 8.0
			)

		line_thickness = 3 if is_selected else 1

		cv2.polylines(
			image,
			[box_int],
			True,
			(255, 255, 255),
			line_thickness,
		)

		text_x = int(candidate.center[0])
		text_y = int(candidate.center[1])

		label = (
			f"#{index} "
			f"S={candidate.score:.2f} "
			f"B={candidate.border_cm:.2f}cm "
			f"R={candidate.ratio:.2f}"
		)

		cv2.putText(
			image,
			label,
			(max(text_x - 150, 5), max(text_y, 20)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.48,
			(255, 255, 255),
			1,
			cv2.LINE_AA,
		)


def warp_frame(frame: np.ndarray, frame_points: np.ndarray) -> np.ndarray:
	target_points = np.array(
		[
			[0, 0],
			[FRAME_WARP_WIDTH - 1, 0],
			[FRAME_WARP_WIDTH - 1, FRAME_WARP_HEIGHT - 1],
			[0, FRAME_WARP_HEIGHT - 1],
		],
		dtype=np.float32,
	)

	matrix = cv2.getPerspectiveTransform(
		order_points(frame_points),
		target_points,
	)

	return cv2.warpPerspective(
		frame,
		matrix,
		(FRAME_WARP_WIDTH, FRAME_WARP_HEIGHT),
	)


def build_color_mask(hsv_image: np.ndarray, ranges: list) -> np.ndarray:
	mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

	for lower, upper in ranges:
		part = cv2.inRange(
			hsv_image,
			np.array(lower, dtype=np.uint8),
			np.array(upper, dtype=np.uint8),
		)
		mask = cv2.bitwise_or(mask, part)

	kernel = cv2.getStructuringElement(
		cv2.MORPH_ELLIPSE,
		(COLOR_MORPH_KERNEL_SIZE, COLOR_MORPH_KERNEL_SIZE),
	)

	mask = cv2.morphologyEx(
		mask,
		cv2.MORPH_OPEN,
		kernel,
		iterations=1,
	)
	mask = cv2.morphologyEx(
		mask,
		cv2.MORPH_CLOSE,
		kernel,
		iterations=2,
	)

	# 中值滤波减少球体边缘因噪声、阴影产生的逐帧毛刺。
	mask = cv2.medianBlur(mask, 5)

	return mask


def classify_shape(
	contour: np.ndarray,
) -> tuple[str, float, float, int]:
	area = cv2.contourArea(contour)
	perimeter = cv2.arcLength(contour, True)

	if perimeter <= 0:
		return "unknown", 0.0, 0.0, 0

	circularity = (
		4.0 * math.pi * area
		/ (perimeter * perimeter)
	)

	rect = cv2.minAreaRect(contour)
	(_, _), (width, height), _ = rect

	if width < 1 or height < 1:
		return "unknown", circularity, 0.0, 0

	long_side = max(width, height)
	short_side = min(width, height)
	aspect_ratio = long_side / short_side

	approx = cv2.approxPolyDP(
		contour,
		APPROX_EPSILON_RATIO * perimeter,
		True,
	)
	vertex_count = len(approx)

	# 球优先判断。
	if (
		circularity >= BALL_CIRCULARITY_THRESHOLD
		and aspect_ratio <= BALL_MAX_RATIO
		and vertex_count >= BALL_MIN_VERTEX_COUNT
	):
		return (
			"ball",
			circularity,
			aspect_ratio,
			vertex_count,
		)

	# Cube/Cuboid 使用迟滞区间：
	# R <= 1.27 明确认定 Cube；
	# R >= 1.33 明确认定 Cuboid；
	# 中间区域交给多帧跟踪保持上一结果。
	if aspect_ratio <= CUBE_MAX_RATIO:
		shape_name = "cube"
	elif aspect_ratio >= CUBOID_MIN_RATIO:
		shape_name = "cuboid"
	else:
		shape_name = "uncertain"

	return (
		shape_name,
		circularity,
		aspect_ratio,
		vertex_count,
	)


def contour_center(contour: np.ndarray) -> tuple[int, int]:
	moments = cv2.moments(contour)

	if abs(moments["m00"]) > 1e-6:
		center_x = int(
			moments["m10"] / moments["m00"]
		)
		center_y = int(
			moments["m01"] / moments["m00"]
		)
		return center_x, center_y

	x, y, width, height = cv2.boundingRect(contour)
	return (
		x + width // 2,
		y + height // 2,
	)


def bounding_box_iou(
	contour_a: np.ndarray,
	contour_b: np.ndarray,
) -> float:
	ax, ay, aw, ah = cv2.boundingRect(contour_a)
	bx, by, bw, bh = cv2.boundingRect(contour_b)

	left = max(ax, bx)
	top = max(ay, by)
	right = min(ax + aw, bx + bw)
	bottom = min(ay + ah, by + bh)

	if right <= left or bottom <= top:
		return 0.0

	intersection = (right - left) * (bottom - top)
	union = aw * ah + bw * bh - intersection

	return intersection / union if union > 0 else 0.0


def remove_duplicate_detections(
	detections: list[Detection],
) -> list[Detection]:
	result = []

	for detection in detections:
		is_duplicate = False

		for accepted in result:
			iou = bounding_box_iou(
				detection.contour,
				accepted.contour,
			)
			distance = math.dist(
				detection.center,
				accepted.center,
			)

			if iou > 0.45 or distance < 25:
				is_duplicate = True
				break

		if not is_duplicate:
			result.append(detection)

	return result


@dataclass
class ObjectTrack:
	track_id: int
	color_name: str
	center: tuple[int, int]
	history: deque
	stable_shape: str | None = None
	missed_frames: int = 0


class ObjectStabilizer:
	def __init__(self) -> None:
		self.tracks: dict[int, ObjectTrack] = {}
		self.next_track_id = 1

	def _new_track(self, detection: Detection) -> ObjectTrack:
		track = ObjectTrack(
			track_id=self.next_track_id,
			color_name=detection.color_name,
			center=detection.center,
			history=deque(maxlen=STABLE_WINDOW),
		)
		self.tracks[track.track_id] = track
		self.next_track_id += 1
		return track

	def _find_track(
		self,
		detection: Detection,
		used_track_ids: set[int],
	) -> ObjectTrack | None:
		best_track = None
		best_distance = float("inf")

		for track in self.tracks.values():
			if track.track_id in used_track_ids:
				continue

			if track.color_name != detection.color_name:
				continue

			distance = math.dist(
				track.center,
				detection.center,
			)

			if (
				distance <= TRACK_MAX_DISTANCE_PX
				and distance < best_distance
			):
				best_track = track
				best_distance = distance

		return best_track

	@staticmethod
	def _fallback_shape(detection: Detection) -> str:
		# 首次出现且刚好落在迟滞区时，用区间中点给出临时结果。
		midpoint = (
			CUBE_MAX_RATIO + CUBOID_MIN_RATIO
		) / 2.0

		if detection.aspect_ratio < midpoint:
			return "cube"

		return "cuboid"

	def update(
		self,
		detections: list[Detection],
	) -> list[Detection]:
		for track in self.tracks.values():
			track.missed_frames += 1

		used_track_ids: set[int] = set()

		for detection in detections:
			track = self._find_track(
				detection,
				used_track_ids,
			)

			if track is None:
				track = self._new_track(detection)

			used_track_ids.add(track.track_id)
			track.center = detection.center
			track.missed_frames = 0

			raw_shape = detection.shape_name

			if raw_shape == "uncertain":
				if track.stable_shape is not None:
					raw_shape = track.stable_shape
				elif track.history:
					raw_shape = Counter(
						track.history
					).most_common(1)[0][0]
				else:
					raw_shape = self._fallback_shape(
						detection
					)

			track.history.append(raw_shape)
			counts = Counter(track.history)
			majority_shape, majority_count = (
				counts.most_common(1)[0]
			)

			if majority_count >= STABLE_MIN_COUNT:
				track.stable_shape = majority_shape
			elif track.stable_shape is None:
				track.stable_shape = majority_shape

			detection.shape_name = (
				track.stable_shape
				if track.stable_shape is not None
				else majority_shape
			)

		expired_ids = [
			track_id
			for track_id, track in self.tracks.items()
			if track.missed_frames
			> TRACK_MAX_MISSED_FRAMES
		]

		for track_id in expired_ids:
			del self.tracks[track_id]

		return detections


def detect_objects(
	image: np.ndarray,
) -> tuple[list[Detection], dict[str, np.ndarray]]:
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	detections = []
	masks = {}

	for color_name, ranges in COLOR_RANGES.items():
		mask = build_color_mask(hsv, ranges)
		masks[color_name] = mask

		contours, _ = cv2.findContours(
			mask,
			cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE,
		)

		for contour in contours:
			area = cv2.contourArea(contour)

			if area < MIN_OBJECT_AREA:
				continue

			(
				shape_name,
				circularity,
				aspect_ratio,
				vertex_count,
			) = classify_shape(contour)
			center = contour_center(contour)

			score = area * max(circularity, 0.2)

			detections.append(
				Detection(
					color_name=color_name,
					shape_name=shape_name,
					contour=contour,
					center=center,
					area=area,
					circularity=circularity,
					aspect_ratio=aspect_ratio,
					vertex_count=vertex_count,
					score=score,
				)
			)

	detections.sort(
		key=lambda item: item.score,
		reverse=True,
	)
	detections = remove_duplicate_detections(
		detections
	)

	return detections[:2], masks


def draw_detection(
	image: np.ndarray,
	detection: Detection,
	index: int,
) -> None:
	contour = detection.contour
	center_x, center_y = detection.center

	cv2.circle(
		image,
		(center_x, center_y),
		5,
		(255, 255, 255),
		-1,
	)

	if detection.shape_name == "ball":
		(circle_x, circle_y), radius = (
			cv2.minEnclosingCircle(contour)
		)
		cv2.circle(
			image,
			(int(circle_x), int(circle_y)),
			int(radius),
			(255, 255, 255),
			2,
		)
	else:
		rect = cv2.minAreaRect(contour)
		box = cv2.boxPoints(rect).astype(np.int32)

		cv2.polylines(
			image,
			[box],
			True,
			(255, 255, 255),
			2,
		)

	title = (
		f"{index}: "
		f"{COLOR_TEXT[detection.color_name]} "
		f"{SHAPE_TEXT[detection.shape_name]}"
	)

	feature_text = (
		f"A={detection.area:.0f} "
		f"C={detection.circularity:.2f} "
		f"R={detection.aspect_ratio:.2f} "
		f"V={detection.vertex_count}"
	)

	text_x = max(center_x - 80, 5)
	text_y = max(center_y - 25, 25)

	cv2.putText(
		image,
		title,
		(text_x, text_y),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.65,
		(255, 255, 255),
		2,
		cv2.LINE_AA,
	)

	cv2.putText(
		image,
		feature_text,
		(text_x, text_y + 24),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.52,
		(255, 255, 255),
		1,
		cv2.LINE_AA,
	)


def create_mask_preview(
	masks: dict[str, np.ndarray],
) -> np.ndarray:
	mask_images = []

	for color_name in COLOR_RANGES:
		mask = masks.get(color_name)

		if mask is None:
			continue

		preview = cv2.cvtColor(
			mask,
			cv2.COLOR_GRAY2BGR,
		)

		cv2.putText(
			preview,
			COLOR_TEXT[color_name],
			(10, 28),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.75,
			(255, 255, 255),
			2,
			cv2.LINE_AA,
		)

		mask_images.append(preview)

	if not mask_images:
		return np.zeros(
			(200, 400, 3),
			dtype=np.uint8,
		)

	target_height = 180
	resized = []

	for image in mask_images:
		scale = target_height / image.shape[0]
		target_width = max(
			1,
			int(image.shape[1] * scale),
		)
		resized.append(
			cv2.resize(
				image,
				(target_width, target_height),
			)
		)

	return np.hstack(resized)


def main() -> None:
	camera = cv2.VideoCapture(
		CAMERA_ID,
		cv2.CAP_DSHOW,
	)

	if not camera.isOpened():
		raise RuntimeError(
			f"无法打开摄像头 {CAMERA_ID}"
		)

	camera.set(
		cv2.CAP_PROP_FRAME_WIDTH,
		CAMERA_WIDTH,
	)
	camera.set(
		cv2.CAP_PROP_FRAME_HEIGHT,
		CAMERA_HEIGHT,
	)

	print(
		"实际分辨率：",
		int(
			camera.get(
				cv2.CAP_PROP_FRAME_WIDTH
			)
		),
		int(
			camera.get(
				cv2.CAP_PROP_FRAME_HEIGHT
			)
		),
	)

	print("按键说明：")
	print("  q：退出")
	print("  c：强制重新寻找黑框")
	print("  w：开启/关闭透视矫正")
	print("  m：显示/隐藏颜色掩膜")
	print("  b：显示/隐藏黑色掩膜")
	print("  f：显示/隐藏所有候选框")

	frame_tracker = FrameTracker()
	object_stabilizer = ObjectStabilizer()
	use_warp = True
	show_masks = False
	show_black_mask = False
	frame_count = 0

	global SHOW_FRAME_CANDIDATES

	while True:
		success, frame = camera.read()

		if not success or frame is None:
			print("读取摄像头画面失败")
			break

		frame_count += 1
		display_source = frame.copy()
		candidates = []
		black_mask = None

		need_update = (
			frame_tracker.points is None
			or frame_count % FRAME_UPDATE_INTERVAL == 0
		)

		if use_warp and need_update:
			candidates, black_mask = (
				find_black_frame_candidates(
					frame,
					frame_tracker.points,
				)
			)

			selected, reacquired = frame_tracker.update(
				candidates
			)

			if reacquired and selected is not None:
				print(
					"黑框捕获/切换成功："
					f"score={selected.score:.2f}, "
					f"border={selected.border_cm:.2f}cm, "
					f"ratio={selected.ratio:.2f}"
				)

		elif show_black_mask or SHOW_FRAME_CANDIDATES:
			candidates, black_mask = (
				find_black_frame_candidates(
					frame,
					frame_tracker.points,
				)
			)

		draw_frame_candidates(
			display_source,
			candidates,
			frame_tracker.points,
		)

		if frame_tracker.points is not None:
			cv2.polylines(
				display_source,
				[
					frame_tracker.points.astype(
						np.int32
					)
				],
				True,
				(255, 255, 255),
				3,
			)

		if use_warp and frame_tracker.points is not None:
			working_image = warp_frame(
				frame,
				frame_tracker.points,
			)
			mode_text = "Mode: frame warp"
		else:
			working_image = frame.copy()
			mode_text = "Mode: full image"

		detections, masks = detect_objects(
			working_image
		)
		detections = object_stabilizer.update(
			detections
		)
		result = working_image.copy()

		for index, detection in enumerate(
			detections,
			start=1,
		):
			draw_detection(
				result,
				detection,
				index,
			)

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
				f"Frame score="
				f"{frame_tracker.candidate.score:.2f} "
				f"border="
				f"{frame_tracker.candidate.border_cm:.2f}cm"
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

		cv2.imshow(
			"source_candidates",
			display_source,
		)
		cv2.imshow(
			"result",
			result,
		)

		if show_masks:
			cv2.imshow(
				"color_masks",
				create_mask_preview(masks),
			)

		if show_black_mask:
			if black_mask is None:
				black_mask = create_black_mask(frame)

			cv2.imshow(
				"black_mask",
				black_mask,
			)

		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break

		if key == ord("c"):
			candidates, black_mask = (
				find_black_frame_candidates(
					frame,
					None,
				)
			)

			frame_tracker.reset()
			selected = frame_tracker.force_acquire(
				candidates
			)

			if selected is None:
				print("本次没有找到可靠黑框")
			else:
				print(
					"强制更新黑框成功："
					f"score={selected.score:.3f}, "
					f"border={selected.border_cm:.2f}cm, "
					f"ratio={selected.ratio:.2f}"
				)

		if key == ord("w"):
			use_warp = not use_warp
			print(
				f"透视矫正："
				f"{'开启' if use_warp else '关闭'}"
			)

		if key == ord("m"):
			show_masks = not show_masks

			if not show_masks:
				try:
					cv2.destroyWindow("color_masks")
				except cv2.error:
					pass

		if key == ord("b"):
			show_black_mask = not show_black_mask

			if not show_black_mask:
				try:
					cv2.destroyWindow("black_mask")
				except cv2.error:
					pass

		if key == ord("f"):
			SHOW_FRAME_CANDIDATES = (
				not SHOW_FRAME_CANDIDATES
			)
			print(
				"候选框显示："
				f"{'开启' if SHOW_FRAME_CANDIDATES else '关闭'}"
			)

	camera.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
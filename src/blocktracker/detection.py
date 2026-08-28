from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectionConfig:
    max_work_width: int = 1600
    min_cell_width: int = 10
    max_cell_width_ratio: float = 0.065
    min_cell_height: int = 9
    max_cell_height_ratio: float = 0.065
    min_cells: int = 7
    min_score: float = 0.30
    group_gap_ratio: float = 0.035


@dataclass(frozen=True)
class CandidateRegion:
    box: Box
    score: float
    cells: tuple[Box, ...]


class CandidateRegionDetector:
    """Detecta paneles de celdas repetidas sin interpretar texto ni números."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()

    def detect(self, image: np.ndarray) -> list[CandidateRegion]:
        if image is None or image.ndim != 3:
            raise ValueError("Se esperaba una imagen BGR de tres canales")
        work, scale = self._resize(image)
        cells = self._find_cells(work)
        groups = self._group_cells(cells, work.shape[:2])
        candidates = [self._score_group(group, work.shape[:2]) for group in groups]
        candidates = [item for item in candidates if item.score >= self.config.min_score]
        candidates.sort(key=lambda item: item.score, reverse=True)
        candidates = self._non_maximum_suppression(candidates)
        if scale != 1.0:
            candidates = [self._restore_scale(item, scale) for item in candidates]
        return candidates

    def _resize(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        width = image.shape[1]
        if width <= self.config.max_work_width:
            return image, 1.0
        scale = self.config.max_work_width / width
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale

    def _find_cells(self, image: np.ndarray) -> list[Box]:
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 45, 135)
        # Los bordes conservan las cajas oscuras/claras sin unirlas con el texto
        # interior, algo que un umbral adaptativo tiende a hacer en paneles densos.
        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        max_w = max(self.config.min_cell_width + 1, int(width * self.config.max_cell_width_ratio))
        max_h = max(self.config.min_cell_height + 1, int(height * self.config.max_cell_height_ratio))
        result: list[Box] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not (self.config.min_cell_width <= w <= max_w and self.config.min_cell_height <= h <= max_h):
                continue
            if not (0.55 <= w / h <= 3.2):
                continue
            contour_area = cv2.contourArea(contour)
            if contour_area < w * h * 0.12:
                continue
            result.append((x, y, w, h))
        return self._deduplicate_boxes(result)

    @staticmethod
    def _deduplicate_boxes(boxes: list[Box]) -> list[Box]:
        boxes = sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)
        kept: list[Box] = []
        for box in boxes:
            if not any(_iou(box, other) > 0.72 or _contains(other, box, 0.86) for other in kept):
                kept.append(box)
        return kept

    def _group_cells(self, cells: list[Box], shape: tuple[int, int]) -> list[list[Box]]:
        if not cells:
            return []
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8)
        for x, y, w, h in cells:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        gap = max(5, int(min(width, height) * self.config.group_gap_ratio))
        joined = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (gap, gap)), iterations=1)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
        groups: list[list[Box]] = []
        centers = [(x + w // 2, y + h // 2) for x, y, w, h in cells]
        for label in range(1, count):
            x, y, w, h, _ = stats[label]
            group = [box for box, (cx, cy) in zip(cells, centers) if x <= cx < x + w and y <= cy < y + h]
            if len(group) >= self.config.min_cells:
                groups.append(group)
        return groups

    def _score_group(self, cells: list[Box], shape: tuple[int, int]) -> CandidateRegion:
        height, width = shape
        x1 = min(box[0] for box in cells)
        y1 = min(box[1] for box in cells)
        x2 = max(box[0] + box[2] for box in cells)
        y2 = max(box[1] + box[3] for box in cells)
        box = (x1, y1, x2 - x1, y2 - y1)
        widths = np.array([item[2] for item in cells], dtype=float)
        heights = np.array([item[3] for item in cells], dtype=float)
        size_consistency = 1.0 - min(1.0, (widths.std() / widths.mean() + heights.std() / heights.mean()) / 1.2)
        row_count = _alignment_count(cells, axis=1)
        column_count = _alignment_count(cells, axis=0)
        alignment = min(1.0, (row_count + column_count) / max(4.0, len(cells) * 0.45))
        count_score = min(1.0, len(cells) / 22.0)
        region_area = max(1, box[2] * box[3])
        density = min(1.0, sum(w * h for _, _, w, h in cells) / region_area * 2.4)
        screen_fraction = region_area / (width * height)
        oversize_penalty = min(0.55, max(0.0, screen_fraction - 0.08) * 2.8)
        large_cell_penalty = min(0.45, max(0.0, np.median(widths) / width - 0.025) * 16)
        score = 0.30 * size_consistency + 0.25 * alignment + 0.25 * count_score + 0.20 * density
        score -= oversize_penalty + large_cell_penalty
        return CandidateRegion(box, float(max(0.0, min(1.0, score))), tuple(cells))

    @staticmethod
    def _non_maximum_suppression(items: list[CandidateRegion]) -> list[CandidateRegion]:
        kept: list[CandidateRegion] = []
        for item in items:
            if not any(_iou(item.box, previous.box) > 0.45 for previous in kept):
                kept.append(item)
        return kept

    @staticmethod
    def _restore_scale(item: CandidateRegion, scale: float) -> CandidateRegion:
        convert = lambda box: tuple(round(value / scale) for value in box)
        return CandidateRegion(convert(item.box), item.score, tuple(convert(cell) for cell in item.cells))  # type: ignore[arg-type]


def _alignment_count(cells: list[Box], axis: int) -> int:
    centers = sorted((box[axis] + box[axis + 2] / 2, box[axis + 2]) for box in cells)
    tolerance = max(3.0, float(np.median([size for _, size in centers])) * 0.45)
    matches = 0
    anchor = centers[0][0]
    group_size = 1
    for center, _ in centers[1:]:
        if center - anchor <= tolerance:
            group_size += 1
        else:
            matches += group_size if group_size >= 2 else 0
            anchor, group_size = center, 1
    return matches + (group_size if group_size >= 2 else 0)


def _contains(outer: Box, inner: Box, threshold: float) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    overlap_w = max(0, min(ox + ow, ix + iw) - max(ox, ix))
    overlap_h = max(0, min(oy + oh, iy + ih) - max(oy, iy))
    return overlap_w * overlap_h >= iw * ih * threshold


def _iou(first: Box, second: Box) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    intersection_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_h = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_w * intersection_h
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0

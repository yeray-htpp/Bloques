from __future__ import annotations

from dataclasses import dataclass

from .detection import Box
from .recognition import RecognizedCell
from .search import SearchResult


@dataclass(frozen=True)
class TrackingUpdate:
    count: int
    box: Box | None
    status: str
    new_result: int | None = None


class LiveHistoryTracker:
    """Sigue snapshots del ticker; no depende de reencontrar solo los 4 iniciales."""

    def __init__(self, initial: SearchResult, target_count: int = 10,
                 confirmations_required: int = 2) -> None:
        self.initial_sequence = tuple(cell.value for cell in initial.match.cells)
        if any(value is None for value in self.initial_sequence):
            raise ValueError("La secuencia inicial debe estar reconocida completamente")
        self.new_values: tuple[int, ...] = ()
        self.current_count = len(self.initial_sequence)
        self.target_count = target_count
        self.confirmations_required = confirmations_required
        self.direction = 0  # 1: entra por izquierda; -1: entra por derecha
        self.last_cells = tuple(initial.recognized)
        self.last_values = _values(self.last_cells)
        self.last_row_box = _enclosing_cells(self.last_cells)
        self.last_block_box = initial.match.enclosing_box
        self.last_anchor_box = initial.match.enclosing_box
        self.pending_values: tuple[int, ...] | None = None
        self.pending_direction = 0
        self.pending_confirmations = 0
        self.lost_frames = 0

    def scan_region(self, image_shape: tuple[int, ...]) -> Box | None:
        """ROI amplia del historial; tras varios fallos vuelve al barrido inferior."""
        screen_h, screen_w = image_shape[:2]
        if self.lost_frames >= 3:
            return 0, 0, screen_w, screen_h
        x, y, w, h = self.last_row_box
        margin_x = max(120, round(screen_w * 0.08))
        margin_y = max(45, round(screen_h * 0.06))
        x1, y1 = max(0, x - margin_x), max(0, y - margin_y)
        x2, y2 = min(screen_w, x + w + margin_x), min(screen_h, y + h + margin_y)
        return x1, y1, x2 - x1, y2 - y1

    def update(self, rows: list[tuple[RecognizedCell, ...]]) -> TrackingUpdate:
        cells = self._select_row(rows)
        if cells is None:
            self.lost_frames += 1
            return TrackingUpdate(self.current_count, self.last_block_box, "lost")
        self.lost_frames = 0
        current_values = _values(cells)
        same_score = _same_score(self.last_values, current_values)
        left_score = _shift_score(self.last_values, current_values, 1)
        right_score = _shift_score(self.last_values, current_values, -1)
        candidate_direction = 1 if left_score >= right_score else -1
        shift_score = max(left_score, right_score)
        is_shift = shift_score >= 0.72 and shift_score >= same_score + 0.12
        new_result: int | None = None
        status = "tracking"

        if is_shift and (self.direction == 0 or candidate_direction == self.direction):
            if (self.pending_values is not None
                    and self.pending_direction == candidate_direction
                    and _same_score(self.pending_values, current_values) >= 0.92):
                self.pending_confirmations += 1
            else:
                self.pending_values = current_values
                self.pending_direction = candidate_direction
                self.pending_confirmations = 1
            status = "confirming"
            if self.pending_confirmations >= self.confirmations_required:
                self.direction = candidate_direction
                new_result = current_values[0] if self.direction == 1 else current_values[-1]
                if self.current_count < self.target_count:
                    if self.direction == 1:
                        self.new_values = (new_result,) + self.new_values
                    else:
                        self.new_values = self.new_values + (new_result,)
                    self.current_count += 1
                self.last_values = current_values
                self.pending_values = None
                self.pending_direction = 0
                self.pending_confirmations = 0
                status = "complete" if self.current_count >= self.target_count else "new_result"
        elif same_score >= 0.72:
            # Mismo snapshot con posibles pequeñas diferencias OCR.
            self.pending_values = None
            self.pending_direction = 0
            self.pending_confirmations = 0
        else:
            # Lectura incoherente: no altera el estado ni cuenta un giro.
            status = "uncertain"

        self.last_cells = cells
        self.last_row_box = _enclosing_cells(cells)
        block_box = self._locate_block(cells)
        if block_box is not None:
            self.last_block_box = block_box
        elif self.current_count >= self.target_count:
            status = "block_not_visible"
        return TrackingUpdate(self.current_count, self.last_block_box, status, new_result)

    def _select_row(self, rows: list[tuple[RecognizedCell, ...]]) -> tuple[RecognizedCell, ...] | None:
        if not rows:
            return None
        old_x, old_y, old_w, old_h = self.last_row_box
        old_cy = old_y + old_h / 2
        best: tuple[float, tuple[RecognizedCell, ...]] | None = None
        for row in rows:
            if len(row) < 4:
                continue
            box = _enclosing_cells(row)
            x, y, w, h = box
            cy = y + h / 2
            vertical = max(0.0, 1.0 - abs(cy - old_cy) / max(20.0, old_h * 5.0))
            overlap = _horizontal_overlap((old_x, old_w), (x, w))
            values = _values(row)
            continuity = max(
                _same_score(self.last_values, values),
                _shift_score(self.last_values, values, 1),
                _shift_score(self.last_values, values, -1),
            )
            contains_anchor = 1.0 if _find_sequence(values, tuple(int(v) for v in self.initial_sequence)) else 0.0
            contains_new = 1.0 if self.new_values and _find_sequence(values, self.new_values) else 0.0
            score = vertical * 2.5 + overlap * 1.5 + continuity * 3.0 + contains_anchor * 3.0 + contains_new * 2.0
            if best is None or score > best[0]:
                best = (score, row)
        return best[1] if best and best[0] >= 2.0 else None

    def _locate_block(self, cells: tuple[RecognizedCell, ...]) -> Box | None:
        values = _values(cells)
        anchors = _find_sequence(values, tuple(int(v) for v in self.initial_sequence))
        if not anchors:
            return None
        old_anchor_center = self.last_anchor_box[0] + self.last_anchor_box[2] / 2
        anchor = min(anchors, key=lambda start: abs(
            _enclosing_cells(cells[start:start + 4])[0]
            + _enclosing_cells(cells[start:start + 4])[2] / 2
            - old_anchor_center
        ))
        anchor_cells = cells[anchor:anchor + 4]
        self.last_anchor_box = _enclosing_cells(anchor_cells)
        if not self.new_values:
            return self.last_anchor_box

        new_occurrences = _find_sequence(values, self.new_values)
        if self.direction == 1:
            new_occurrences = [start for start in new_occurrences if start + len(self.new_values) <= anchor]
            if new_occurrences:
                new_start = max(new_occurrences)
            else:
                return None
        else:
            new_occurrences = [start for start in new_occurrences if start >= anchor + 4]
            if new_occurrences:
                new_start = min(new_occurrences)
            else:
                return None
        new_cells = cells[new_start:new_start + len(self.new_values)]
        selected = tuple(new_cells) + tuple(anchor_cells)
        if len(selected) != self.current_count:
            return None
        return _enclosing_cells(selected)


def _values(cells: tuple[RecognizedCell, ...]) -> tuple[int, ...]:
    return tuple(int(cell.value) for cell in cells if cell.value is not None)


def _same_score(first: tuple[int, ...], second: tuple[int, ...]) -> float:
    compared = min(len(first), len(second))
    if compared < 4:
        return 0.0
    return sum(a == b for a, b in zip(first[:compared], second[:compared])) / compared


def _shift_score(previous: tuple[int, ...], current: tuple[int, ...], direction: int) -> float:
    if direction == 1:  # nuevo a la izquierda
        compared = min(len(previous), max(0, len(current) - 1))
        pairs = zip(previous[:compared], current[1:1 + compared])
    else:  # nuevo a la derecha; los anteriores se desplazan a la izquierda
        compared = min(max(0, len(previous) - 1), len(current))
        pairs = zip(previous[1:1 + compared], current[:compared])
    if compared < 4:
        return 0.0
    return sum(a == b for a, b in pairs) / compared


def _find_sequence(values: tuple[int, ...], sequence: tuple[int, ...]) -> list[int]:
    if not sequence:
        return []
    return [index for index in range(len(values) - len(sequence) + 1)
            if values[index:index + len(sequence)] == sequence]


def _horizontal_overlap(first: tuple[int, int], second: tuple[int, int]) -> float:
    ax, aw = first
    bx, bw = second
    overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    return overlap / max(1, min(aw, bw))


def _enclosing_cells(cells: tuple[RecognizedCell, ...]) -> Box:
    x1 = min(cell.box[0] for cell in cells)
    y1 = min(cell.box[1] for cell in cells)
    x2 = max(cell.box[0] + cell.box[2] for cell in cells)
    y2 = max(cell.box[1] + cell.box[3] for cell in cells)
    return x1, y1, x2 - x1, y2 - y1

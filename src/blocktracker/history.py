from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detection import Box
from .recognition import RecognizedCell


@dataclass(frozen=True)
class SequenceMatch:
    sequence: tuple[int, ...]
    cells: tuple[RecognizedCell, ...]
    enclosing_box: Box


class HistoryParser:
    """Orden visual desacoplado: filas de arriba abajo y X de izquierda a derecha."""

    def parse(self, cells: list[RecognizedCell]) -> list[RecognizedCell]:
        if not cells:
            return []
        tolerance = max(3.0, float(np.median([cell.box[3] for cell in cells])) * 0.55)
        rows: list[list[RecognizedCell]] = []
        for cell in sorted(cells, key=lambda item: item.box[1] + item.box[3] / 2):
            center_y = cell.box[1] + cell.box[3] / 2
            target = next((row for row in rows if abs(center_y - _row_center(row)) <= tolerance), None)
            if target is None:
                rows.append([cell])
            else:
                target.append(cell)
        rows.sort(key=_row_center)
        return [cell for row in rows for cell in sorted(row, key=lambda item: item.box[0])]


class SequenceFinder:
    def find(self, cells: list[RecognizedCell], sequence: tuple[int, ...]) -> SequenceMatch | None:
        for start in range(len(cells) - len(sequence) + 1):
            window = cells[start:start + len(sequence)]
            if tuple(cell.value for cell in window) == sequence:
                return SequenceMatch(sequence, tuple(window), _enclosing_box(window))
        return None


def _row_center(row: list[RecognizedCell]) -> float:
    return float(np.mean([cell.box[1] + cell.box[3] / 2 for cell in row]))


def _enclosing_box(cells: list[RecognizedCell]) -> Box:
    x1 = min(cell.box[0] for cell in cells)
    y1 = min(cell.box[1] for cell in cells)
    x2 = max(cell.box[0] + cell.box[2] for cell in cells)
    y2 = max(cell.box[1] + cell.box[3] for cell in cells)
    return x1, y1, x2 - x1, y2 - y1

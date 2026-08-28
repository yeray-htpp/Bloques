from __future__ import annotations

import re
import logging
from typing import Any

import cv2
import numpy as np
from rapidocr import RapidOCR

from .detection import CandidateRegion
from .history import SequenceFinder
from .recognition import RecognizedCell
from .search import SearchResult
from .detection import Box


class RapidHistoryRecognizer:
    """OCR ONNX local para texto diminuto; no realiza ninguna conexión externa."""

    def __init__(self, scale: float = 2.5) -> None:
        logging.getLogger("RapidOCR").setLevel(logging.ERROR)
        self.scale = scale
        self.engine = RapidOCR()
        self.finder = SequenceFinder()

    def search(self, image: np.ndarray, sequence: tuple[int, ...]) -> SearchResult | None:
        height, width = image.shape[:2]
        rows = self.read_rows(image)
        matches = self._matches(rows, sequence, width, height)
        if not matches:
            # La zona inferior es la ruta rápida habitual. Si no aparece allí,
            # completa el barrido con la parte superior de la pantalla.
            upper = self.read_rows(image, (0, 0, width, round(height * 0.58)))
            matches = self._matches(upper, sequence, width, height)
        return max(matches, key=lambda item: item[0])[1] if matches else None

    def _matches(self, rows: list[tuple[RecognizedCell, ...]], sequence: tuple[int, ...],
                 width: int, height: int) -> list[tuple[float, SearchResult]]:
        matches: list[tuple[float, SearchResult]] = []
        for cells in rows:
            match = self.finder.find(cells, sequence)
            if not match:
                continue
            x1 = min(cell.box[0] for cell in cells)
            y1 = min(cell.box[1] for cell in cells)
            x2 = max(cell.box[0] + cell.box[2] for cell in cells)
            y2 = max(cell.box[1] + cell.box[3] for cell in cells)
            region = CandidateRegion((x1, y1, x2 - x1, y2 - y1), 0.9, tuple(cell.box for cell in cells))
            result = SearchResult(region, tuple(cells), match)
            center_x = (x1 + x2) / 2 / width
            score = (y1 / height) * 0.65 + abs(center_x - 0.5) * 0.35
            matches.append((score, result))
        return matches

    def read_rows(self, image: np.ndarray, region: Box | None = None) -> list[tuple[RecognizedCell, ...]]:
        """Devuelve snapshots completos. Tras fijar el historial puede limitarse a su ROI."""
        height, width = image.shape[:2]
        if region is None:
            x1, y1, x2, y2 = 0, round(height * 0.52), width, height
        else:
            x, y, w, h = region
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
        if x2 <= x1 or y2 <= y1:
            return []
        crop = image[y1:y2, x1:x2]
        enlarged = cv2.resize(crop, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        output = self.engine(enlarged, return_word_box=True, text_score=0.35, box_thresh=0.35)
        rows = self._rows_from_output(output, x1, y1, width, height)
        return [tuple(row) for row in rows]

    def _rows_from_output(self, output: Any, offset_x: int, offset_y: int,
                          width: int, height: int) -> list[list[RecognizedCell]]:
        all_cells: list[RecognizedCell] = []
        word_rows = getattr(output, "word_results", ()) or ()
        for words in word_rows:
            for word in words:
                text, confidence, points = word
                number_match = re.fullmatch(r"\D*(\d{1,2})\D*", str(text))
                if not number_match:
                    continue
                value = int(number_match.group(1))
                if not 0 <= value <= 36:
                    continue
                xs = [float(point[0]) / self.scale + offset_x for point in points]
                ys = [float(point[1]) / self.scale + offset_y for point in points]
                x1, y1 = max(0, round(min(xs))), max(0, round(min(ys)))
                x2, y2 = min(width, round(max(xs))), min(height, round(max(ys)))
                all_cells.append(RecognizedCell((x1, y1, max(1, x2 - x1), max(1, y2 - y1)), value, float(confidence)))
        # RapidOCR puede devolver toda la fila como una línea o cada color como
        # una línea separada. Reagrupa siempre por coordenada Y.
        rows: list[list[RecognizedCell]] = []
        for cell in sorted(all_cells, key=lambda item: item.box[1] + item.box[3] / 2):
            center_y = cell.box[1] + cell.box[3] / 2
            target = None
            for row in rows:
                row_center = float(np.median([item.box[1] + item.box[3] / 2 for item in row]))
                row_height = float(np.median([item.box[3] for item in row]))
                if abs(center_y - row_center) <= max(3.0, row_height * 0.65):
                    target = row
                    break
            if target is None:
                rows.append([cell])
            else:
                target.append(cell)
        return [sorted(row, key=lambda item: item.box[0]) for row in rows if len(row) >= 4]

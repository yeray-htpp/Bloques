from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detection import Box


@dataclass(frozen=True)
class UnboxedRow:
    box: Box
    tokens: tuple[Box, ...]


class UnboxedHistoryDetector:
    """Encuentra filas de números pequeños aunque no tengan casillas dibujadas."""

    def detect(self, image: np.ndarray) -> list[UnboxedRow]:
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Incluye texto blanco y texto coloreado brillante sobre panel oscuro.
        bright = gray >= 135
        colored = (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 105)
        foreground = ((bright | colored) * 255).astype(np.uint8)
        components = self._glyph_components(foreground, width, height)
        rows = self._cluster_rows(components)
        candidates: list[UnboxedRow] = []
        for glyphs in rows:
            tokens = self._tokens_from_glyphs(glyphs)
            if len(tokens) < 6:
                continue
            token_heights = np.array([box[3] for box in tokens], dtype=float)
            if token_heights.std() / max(1.0, token_heights.mean()) > 0.42:
                continue
            x1 = min(box[0] for box in tokens)
            y1 = min(box[1] for box in tokens)
            x2 = max(box[0] + box[2] for box in tokens)
            y2 = max(box[1] + box[3] for box in tokens)
            # Un historial típico es una franja; evita bloques de texto dispersos.
            if x2 - x1 < width * 0.08:
                continue
            candidates.append(UnboxedRow((x1, y1, x2 - x1, y2 - y1), tuple(tokens)))
        candidates.sort(key=lambda row: len(row.tokens), reverse=True)
        return candidates

    @staticmethod
    def _glyph_components(mask: np.ndarray, width: int, height: int) -> list[Box]:
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        glyphs: list[Box] = []
        min_h = max(4, round(height * 0.004))
        max_h = max(22, round(height * 0.032))
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            if min_h <= h <= max_h and 1 <= w <= h * 1.15 and area >= max(3, h * 0.12):
                glyphs.append((int(x), int(y), int(w), int(h)))
        return glyphs

    @staticmethod
    def _cluster_rows(glyphs: list[Box]) -> list[list[Box]]:
        rows: list[list[Box]] = []
        for glyph in sorted(glyphs, key=lambda box: box[1] + box[3] / 2):
            cy = glyph[1] + glyph[3] / 2
            target = None
            for row in rows:
                median_h = float(np.median([box[3] for box in row]))
                row_y = float(np.median([box[1] + box[3] / 2 for box in row]))
                if abs(cy - row_y) <= max(2.0, median_h * 0.42):
                    target = row
                    break
            if target is None:
                rows.append([glyph])
            else:
                target.append(glyph)
        return rows

    @staticmethod
    def _tokens_from_glyphs(glyphs: list[Box]) -> list[Box]:
        glyphs = sorted(glyphs, key=lambda box: box[0])
        if not glyphs:
            return []
        median_h = float(np.median([box[3] for box in glyphs]))
        tokens: list[list[Box]] = [[glyphs[0]]]
        for glyph in glyphs[1:]:
            previous = tokens[-1][-1]
            gap = glyph[0] - (previous[0] + previous[2])
            # Dentro de un número de dos cifras el espacio es pequeño.
            if gap <= max(1.0, median_h * 0.55) and len(tokens[-1]) < 2:
                tokens[-1].append(glyph)
            else:
                tokens.append([glyph])
        boxes: list[Box] = []
        for token in tokens:
            x1 = min(box[0] for box in token)
            y1 = min(box[1] for box in token)
            x2 = max(box[0] + box[2] for box in token)
            y2 = max(box[1] + box[3] for box in token)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes

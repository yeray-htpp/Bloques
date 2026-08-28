from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detection import Box


@dataclass(frozen=True)
class RecognizedCell:
    box: Box
    value: int | None
    confidence: float


class NumberRecognizer:
    """OCR especializado: reconoce únicamente uno o dos dígitos que formen 0..36."""

    _SIZE = (20, 30)

    def __init__(self, min_confidence: float = 0.43) -> None:
        self.min_confidence = min_confidence
        self._templates = self._build_templates()

    def recognize(self, image: np.ndarray, box: Box) -> RecognizedCell:
        x, y, w, h = box
        pad_x, pad_y = max(1, round(w * 0.10)), max(1, round(h * 0.10))
        roi = image[y + pad_y:y + h - pad_y, x + pad_x:x + w - pad_x]
        return self._recognize_roi(roi, box)

    def recognize_unboxed(self, image: np.ndarray, box: Box) -> RecognizedCell:
        """Lee texto sin marco; conserva todo el glifo detectado."""
        x, y, w, h = box
        margin = max(1, round(h * 0.18))
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(image.shape[1], x + w + margin), min(image.shape[0], y + h + margin)
        return self._recognize_roi(image[y1:y2, x1:x2], box)

    def _recognize_roi(self, roi: np.ndarray, box: Box) -> RecognizedCell:
        if roi.size == 0:
            return RecognizedCell(box, None, 0.0)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, base = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        attempts = [base, cv2.bitwise_not(base)]
        best_value: int | None = None
        best_confidence = 0.0
        for binary in attempts:
            result = self._read_binary(binary)
            if result and result[1] > best_confidence:
                best_value, best_confidence = result
        if best_confidence < self.min_confidence:
            best_value = None
        return RecognizedCell(box, best_value, best_confidence)

    def _read_binary(self, binary: np.ndarray) -> tuple[int, float] | None:
        # El texto debe ser el primer plano minoritario; descarta la polaridad que
        # convierte el fondo completo de la celda en un glifo.
        if cv2.countNonZero(binary) / binary.size > 0.48:
            return None
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = binary.shape
        glyphs: list[tuple[int, np.ndarray]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < height * 0.32 or h > height * 0.96:
                continue
            if w < 2 or w > width * 0.82 or w * h < binary.size * 0.015:
                continue
            glyphs.append((x, binary[y:y + h, x:x + w]))
        glyphs.sort(key=lambda item: item[0])
        if not 1 <= len(glyphs) <= 2:
            return None
        digits: list[str] = []
        scores: list[float] = []
        for _, glyph in glyphs:
            normalized = self._normalize(glyph)
            digit, score = self._match_digit(normalized)
            digits.append(str(digit))
            scores.append(score)
        if len(digits) == 2 and digits[0] == "0":
            return None
        value = int("".join(digits))
        if not 0 <= value <= 36:
            return None
        return value, float(np.mean(scores))

    def _match_digit(self, glyph: np.ndarray) -> tuple[int, float]:
        best_digit, best_score = 0, -1.0
        glyph_float = glyph.astype(np.float32) / 255.0
        glyph_holes = self._hole_count(glyph)
        for digit, variants in self._templates.items():
            for template in variants:
                score = float(cv2.matchTemplate(glyph_float, template, cv2.TM_CCOEFF_NORMED)[0, 0])
                template_holes = self._hole_count((template * 255).astype(np.uint8))
                score -= abs(glyph_holes - template_holes) * 0.16
                if score > best_score:
                    best_digit, best_score = digit, score
        return best_digit, max(0.0, best_score)

    @staticmethod
    def _hole_count(glyph: np.ndarray) -> int:
        contours, hierarchy = cv2.findContours(glyph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return 0
        return sum(1 for item in hierarchy[0] if item[3] >= 0)

    @classmethod
    def _normalize(cls, glyph: np.ndarray) -> np.ndarray:
        points = cv2.findNonZero(glyph)
        if points is None:
            return np.zeros((cls._SIZE[1], cls._SIZE[0]), dtype=np.uint8)
        x, y, w, h = cv2.boundingRect(points)
        glyph = glyph[y:y + h, x:x + w]
        target_w, target_h = cls._SIZE
        scale = min((target_w - 4) / max(1, w), (target_h - 4) / max(1, h))
        resized = cv2.resize(glyph, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        ox, oy = (target_w - resized.shape[1]) // 2, (target_h - resized.shape[0]) // 2
        canvas[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
        return canvas

    @classmethod
    def _build_templates(cls) -> dict[int, list[np.ndarray]]:
        templates: dict[int, list[np.ndarray]] = {digit: [] for digit in range(10)}
        fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN]
        for digit in range(10):
            for font in fonts:
                for scale in (0.75, 0.9, 1.05):
                    for thickness in (1, 2):
                        canvas = np.zeros((48, 36), dtype=np.uint8)
                        size, _ = cv2.getTextSize(str(digit), font, scale, thickness)
                        origin = ((36 - size[0]) // 2, (48 + size[1]) // 2)
                        cv2.putText(canvas, str(digit), origin, font, scale, 255, thickness, cv2.LINE_AA)
                        _, canvas = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        normalized = cls._normalize(canvas).astype(np.float32) / 255.0
                        templates[digit].append(normalized)
        return templates

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detection import Box, CandidateRegion, CandidateRegionDetector
from .history import HistoryParser, SequenceFinder, SequenceMatch
from .recognition import NumberRecognizer, RecognizedCell
from .unboxed import UnboxedHistoryDetector


@dataclass(frozen=True)
class SearchResult:
    region: CandidateRegion
    recognized: tuple[RecognizedCell, ...]
    match: SequenceMatch


class ScreenSequenceSearcher:
    def __init__(self) -> None:
        self.detector = CandidateRegionDetector()
        self.recognizer = NumberRecognizer()
        self.parser = HistoryParser()
        self.finder = SequenceFinder()
        self.unboxed_detector = UnboxedHistoryDetector()
        self._rapid_recognizer = None

    def search(self, image: np.ndarray, sequence: tuple[int, ...]) -> SearchResult | None:
        # El OCR robusto es la fuente de verdad. Las heurísticas antiguas podían
        # confundir números del paño central con el historial.
        if self._rapid_recognizer is None:
            from .rapid_recognition import RapidHistoryRecognizer

            self._rapid_recognizer = RapidHistoryRecognizer()
        return self._rapid_recognizer.search(image, sequence)

    def read_history_rows(self, image: np.ndarray, region: Box | None = None) -> list[tuple[RecognizedCell, ...]]:
        if self._rapid_recognizer is None:
            from .rapid_recognition import RapidHistoryRecognizer

            self._rapid_recognizer = RapidHistoryRecognizer()
        return self._rapid_recognizer.read_rows(image, region)

    def search_legacy(self, image: np.ndarray, sequence: tuple[int, ...]) -> SearchResult | None:
        """Detector experimental conservado para diagnóstico, no usado en vivo."""
        matches: list[tuple[float, SearchResult]] = []
        screen_h, screen_w = image.shape[:2]
        # Primero evalúa historiales sin casillas (formato de la captura de Betano).
        for row in self.unboxed_detector.detect(image):
            recognized = [self.recognizer.recognize_unboxed(image, token) for token in row.tokens]
            match = self.finder.find(recognized, sequence)
            if match:
                region = CandidateRegion(row.box, 0.50, row.tokens)
                result = SearchResult(region, tuple(recognized), match)
                matches.append((self._location_score(row.box, screen_w, screen_h, unboxed=True), result))
        for region in self.detector.detect(image):
            recognized = [self.recognizer.recognize(image, cell) for cell in region.cells]
            ordered = self.parser.parse(recognized)
            match = self.finder.find(ordered, sequence)
            if match:
                result = SearchResult(region, tuple(ordered), match)
                matches.append((self._location_score(region.box, screen_w, screen_h, unboxed=False), result))
        return max(matches, key=lambda item: item[0])[1] if matches else None

    @staticmethod
    def _location_score(box: Box, screen_w: int, screen_h: int, unboxed: bool) -> float:
        x, y, w, h = box
        center_x = (x + w / 2) / screen_w
        center_y = (y + h / 2) / screen_h
        edge = abs(center_x - 0.5) * 2.0
        lower = center_y
        row_shape = min(1.0, w / max(1.0, h * 5))
        # Es una preferencia, no una exclusión: abajo y cerca de un borde coincide
        # con los historiales reales mostrados, mientras el paño queda al centro.
        return lower * 0.42 + edge * 0.28 + row_shape * 0.15 + (0.15 if unboxed else 0.0)


def translate_box(box: Box, origin: tuple[int, int]) -> Box:
    return box[0] + origin[0], box[1] + origin[1], box[2], box[3]


def match_index(result: SearchResult) -> int:
    first_box = result.match.cells[0].box
    return next(index for index, cell in enumerate(result.recognized) if cell.box == first_box)

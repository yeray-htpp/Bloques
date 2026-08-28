from __future__ import annotations

import cv2
import numpy as np

from .detection import CandidateRegion


class DebugRenderer:
    @staticmethod
    def render(image: np.ndarray, candidates: list[CandidateRegion], show_all: bool = False) -> np.ndarray:
        output = image.copy()
        selected = candidates if show_all else candidates[:1]
        for index, candidate in enumerate(selected, start=1):
            x, y, w, h = candidate.box
            color = (40, 220, 40) if index == 1 else (0, 180, 255)
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 3)
            label_y = max(24, y - 8)
            cv2.putText(output, f"historial candidato {index}: {candidate.score:.2f}", (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            for cx, cy, cw, ch in candidate.cells:
                cv2.rectangle(output, (cx, cy), (cx + cw, cy + ch), (220, 120, 30), 1)
        return output


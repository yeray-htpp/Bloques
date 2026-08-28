from __future__ import annotations

from pathlib import Path

import cv2
import mss
import numpy as np


class ScreenCapture:
    """Obtiene píxeles visibles sin interactuar con navegador, DOM o controles."""

    def capture(self, monitor: int = 0) -> np.ndarray:
        image, _ = self.capture_with_origin(monitor)
        return image

    def capture_with_origin(self, monitor: int = 0) -> tuple[np.ndarray, tuple[int, int]]:
        with mss.mss() as grabber:
            if monitor < 0 or monitor >= len(grabber.monitors):
                raise ValueError(f"Monitor inválido: {monitor}")
            description = grabber.monitors[monitor]
            shot = np.asarray(grabber.grab(description))
        image = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
        return image, (description["left"], description["top"])

    @staticmethod
    def load(path: str | Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"No se pudo leer la imagen: {path}")
        return image

import cv2
import numpy as np

from blocktracker.detection import CandidateRegionDetector, DetectionConfig


def _scene() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    image = np.full((720, 1280, 3), 25, dtype=np.uint8)
    # Historial compacto: muchas celdas pequeñas y regulares.
    expected = (55, 500, 300, 100)
    for row in range(4):
        for column in range(12):
            x, y = 55 + column * 25, 500 + row * 25
            cv2.rectangle(image, (x, y), (x + 22, y + 21), (130, 130, 130), 1)
            cv2.putText(image, str((row * 12 + column) % 37), (x + 3, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (235, 235, 235), 1)
    # Mesa de apuestas: celdas deliberadamente grandes.
    for row in range(3):
        for column in range(12):
            x, y = 400 + column * 65, 270 + row * 52
            cv2.rectangle(image, (x, y), (x + 62, y + 48), (180, 180, 180), 2)
    return image, expected


def test_compact_history_is_detected_ahead_of_betting_table() -> None:
    image, expected = _scene()
    detector = CandidateRegionDetector(DetectionConfig(min_score=0.20))
    candidates = detector.detect(image)
    assert candidates
    x, y, w, h = candidates[0].box
    ex, ey, ew, eh = expected
    assert x < ex + ew and x + w > ex
    assert y < ey + eh and y + h > ey


def test_invalid_image_is_rejected() -> None:
    detector = CandidateRegionDetector()
    try:
        detector.detect(np.zeros((10, 10), dtype=np.uint8))
    except ValueError:
        pass
    else:
        raise AssertionError("Debió rechazar una imagen sin tres canales")

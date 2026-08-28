import cv2
import numpy as np

from blocktracker.history import HistoryParser, SequenceFinder
from blocktracker.recognition import NumberRecognizer, RecognizedCell
from blocktracker.search import ScreenSequenceSearcher
from blocktracker.search import SearchResult
from blocktracker.detection import CandidateRegion
from blocktracker.history import SequenceMatch
from blocktracker.tracking import LiveHistoryTracker


def _number_cell(value: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    image = np.full((70, 90, 3), 28, dtype=np.uint8)
    box = (10, 10, 70, 50)
    cv2.rectangle(image, (10, 10), (80, 60), (120, 120, 120), 1)
    text = str(value)
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(image, text, (45 - size[0] // 2, 35 + size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2, cv2.LINE_AA)
    return image, box


def test_recognizer_is_limited_to_roulette_numbers() -> None:
    recognizer = NumberRecognizer(min_confidence=0.35)
    for expected in (0, 6, 21, 30, 36):
        image, box = _number_cell(expected)
        result = recognizer.recognize(image, box)
        assert result.value == expected, (expected, result)


def test_parser_and_finder_follow_visual_rows() -> None:
    cells = [
        RecognizedCell((60, 40, 20, 20), 27, 0.9),
        RecognizedCell((10, 70, 20, 20), 30, 0.9),
        RecognizedCell((10, 40, 20, 20), 21, 0.9),
        RecognizedCell((60, 70, 20, 20), 6, 0.9),
    ]
    ordered = HistoryParser().parse(cells)
    match = SequenceFinder().find(ordered, (21, 27, 30, 6))
    assert [cell.value for cell in ordered] == [21, 27, 30, 6]
    assert match is not None
    assert match.enclosing_box == (10, 40, 70, 50)


def test_search_finds_small_unboxed_betano_style_history() -> None:
    image = np.full((300, 800, 3), 12, dtype=np.uint8)
    values = [32, 6, 31, 26, 10, 5, 8, 31, 33, 20, 5]
    x, baseline = 360, 260
    for index, value in enumerate(values):
        color = (235, 235, 235) if index % 3 else (40, 40, 235)
        cv2.putText(image, str(value), (x, baseline), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
        x += 36
    result = ScreenSequenceSearcher().search(image, (8, 31, 33, 20))
    assert result is not None
    assert tuple(cell.value for cell in result.match.cells) == (8, 31, 33, 20)


def _tracking_result(start: int, prefix: list[int] | None = None,
                     x_offset: int = 0) -> SearchResult:
    prefix = prefix or []
    values = prefix + [8, 31, 33, 20] + [11, 12, 14, 15, 16, 17, 18, 21, 22, 23, 24]
    cells = tuple(RecognizedCell((10 + i * 20, 40, 16, 12), value, 0.9)
                  for i, value in enumerate(values))
    if x_offset:
        cells = tuple(RecognizedCell((cell.box[0] + x_offset, *cell.box[1:]), cell.value, cell.confidence)
                      for cell in cells)
    matched = cells[start:start + 4]
    match = SequenceMatch((8, 31, 33, 20), matched, (matched[0].box[0], 40, 76, 12))
    return SearchResult(CandidateRegion((cells[0].box[0], 40, len(cells) * 20, 12), 0.8,
                                        tuple(c.box for c in cells)), cells, match)


def test_block_progress_counts_sequence_displacement_to_ten() -> None:
    tracker = LiveHistoryTracker(_tracking_result(0))
    current_prefix: list[int] = []
    for added in range(1, 7):
        current_prefix.insert(0, added)
        current = _tracking_result(added, current_prefix)
        # Dos snapshots iguales confirman una inserción; uno solo puede ser ruido.
        assert tracker.update([current.recognized]).count == 3 + added
        update = tracker.update([current.recognized])
        assert update.count == 4 + added
        assert update.box is not None and update.box[2] > 0
    assert tracker.current_count == 10


def test_completed_block_keeps_moving_without_counting_more() -> None:
    tracker = LiveHistoryTracker(_tracking_result(0))
    prefix: list[int] = []
    for value in (1, 2, 3, 4, 5, 6):
        prefix.insert(0, value)
        row = _tracking_result(len(prefix), prefix).recognized
        tracker.update([row])
        tracker.update([row])
    previous_box = tracker.last_block_box
    prefix.insert(0, 7)
    moved = _tracking_result(len(prefix), prefix, x_offset=9).recognized
    tracker.update([moved])
    update = tracker.update([moved])
    assert update.count == 10
    assert update.box is not None and update.box != previous_box


def test_layout_motion_without_new_spin_moves_box_but_not_count() -> None:
    initial = _tracking_result(0)
    tracker = LiveHistoryTracker(initial)
    moved = _tracking_result(0, x_offset=25).recognized
    update = tracker.update([moved])
    assert update.count == 4
    assert update.box is not None
    assert update.box[0] == initial.match.enclosing_box[0] + 25


def test_initial_sequence_in_middle_tracks_future_prefix_separately() -> None:
    initial = _tracking_result(2, [29, 18])
    tracker = LiveHistoryTracker(initial)
    current = _tracking_result(3, [7, 29, 18]).recognized
    tracker.update([current])
    update = tracker.update([current])
    assert update.count == 5
    assert tracker.new_values == (7,)
    assert update.box is not None
    # El borde exterior incluye el resultado nuevo y los cuatro iniciales.
    assert update.box[0] == current[0].box[0]
    assert update.box[0] + update.box[2] == current[6].box[0] + current[6].box[2]

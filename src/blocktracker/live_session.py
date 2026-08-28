from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Thread

from .capture import ScreenCapture
from .detection import Box
from .search import ScreenSequenceSearcher, translate_box
from .tracking import LiveHistoryTracker, TrackingUpdate


@dataclass(frozen=True)
class RenderEvent:
    update: TrackingUpdate | None
    screen_box: Box | None
    error: str | None = None


class LiveTrackingWorker:
    """Realiza captura/OCR fuera del hilo gráfico y publica el último estado."""

    def __init__(self, capture: ScreenCapture, searcher: ScreenSequenceSearcher,
                 tracker: LiveHistoryTracker, monitor: int, interval: float) -> None:
        self.capture = capture
        self.searcher = searcher
        self.tracker = tracker
        self.monitor = monitor
        self.interval = max(0.15, interval)
        self.events: Queue[RenderEvent] = Queue(maxsize=1)
        self.stop_event = Event()
        self.thread = Thread(target=self._run, name="blocktracker-ocr", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3.0)

    def latest(self) -> RenderEvent | None:
        latest_event = None
        while True:
            try:
                latest_event = self.events.get_nowait()
            except Empty:
                return latest_event

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                image, origin = self.capture.capture_with_origin(self.monitor)
                rows = self.searcher.read_history_rows(image, self.tracker.scan_region(image.shape))
                update = self.tracker.update(rows)
                screen_box = translate_box(update.box, origin) if update.box else None
                self._publish(RenderEvent(update, screen_box))
            except Exception as error:
                self._publish(RenderEvent(None, None, str(error)))
            self.stop_event.wait(self.interval)

    def _publish(self, event: RenderEvent) -> None:
        try:
            self.events.put_nowait(event)
        except Full:
            try:
                self.events.get_nowait()
            except Empty:
                pass
            self.events.put_nowait(event)

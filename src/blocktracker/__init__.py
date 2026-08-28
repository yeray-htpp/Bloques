"""BlockTracker: análisis visual pasivo de historiales de ruleta."""

from .capture import ScreenCapture
from .detection import CandidateRegion, CandidateRegionDetector, DetectionConfig
from .history import HistoryParser, SequenceFinder
from .recognition import NumberRecognizer
from .search import ScreenSequenceSearcher
from .unboxed import UnboxedHistoryDetector
from .tracking import LiveHistoryTracker, TrackingUpdate
from .rapid_recognition import RapidHistoryRecognizer

__all__ = [
    "CandidateRegion",
    "CandidateRegionDetector",
    "DetectionConfig",
    "ScreenCapture",
    "HistoryParser",
    "NumberRecognizer",
    "ScreenSequenceSearcher",
    "SequenceFinder",
    "UnboxedHistoryDetector",
    "LiveHistoryTracker",
    "TrackingUpdate",
    "RapidHistoryRecognizer",
]

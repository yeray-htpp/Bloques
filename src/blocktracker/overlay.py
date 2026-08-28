from __future__ import annotations

import ctypes
from ctypes import wintypes
import signal
import sys

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from .detection import Box


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
_user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
_user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_int, wintypes.UINT]
_user32.SetWindowPos.restype = wintypes.BOOL


class _BorderWidget(QWidget):
    def __init__(self, color: str, border: int) -> None:
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        super().__init__(None, flags)
        self.border = border
        self.color = QColor(color)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

    def set_color(self, color: str) -> None:
        self.color = QColor(color)
        self.update()

    def paintEvent(self, event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QPen(self.color, self.border, Qt.PenStyle.SolidLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        half = self.border / 2
        painter.drawRect(QRectF(half, half, self.width() - self.border, self.height() - self.border))
        painter.end()


class OverlayWindow:
    """Overlay Qt transparente, always-on-top y completamente click-through."""

    BORDER = 4

    def __init__(self, box: Box, count: int = 4, target: int = 10,
                 active_color: str = "#00E5FF", complete_color: str = "#FF3CAC") -> None:
        enable_dpi_awareness()
        self.app = QApplication.instance() or QApplication(sys.argv[:1])
        self.active_color = active_color
        self.complete_color = complete_color
        self.widget = _BorderWidget(active_color, self.BORDER)
        self._capture_excluded = False
        self.update(box, count, target)
        # Un temporizador frecuente permite que Python procese Ctrl+C.
        self._signal_timer = QTimer()
        self._signal_timer.timeout.connect(lambda: None)
        self._signal_timer.start(100)
        signal.signal(signal.SIGINT, lambda *_: self.app.quit())

    def update(self, box: Box, count: int, target: int = 10) -> None:
        x, y, w, h = box
        border = self.BORDER
        color = self.active_color if count < target else self.complete_color
        self.widget.set_color(color)
        self.widget.show()
        hwnd = int(self.widget.winId())
        flags = 0x0010 | 0x0040  # SWP_NOACTIVATE | SWP_SHOWWINDOW
        _user32.SetWindowPos(
            hwnd, wintypes.HWND(-1), x - border, y - border,
            max(border * 2, w + border * 2), max(border * 2, h + border * 2), flags
        )
        if not self._capture_excluded:
            self._capture_excluded = bool(_user32.SetWindowDisplayAffinity(hwnd, 0x11))
        self.widget.update()

    def after(self, milliseconds: int, callback: object) -> None:
        QTimer.singleShot(milliseconds, callback)  # type: ignore[arg-type]

    def run(self) -> None:
        self.app.exec()


def enable_dpi_awareness() -> None:
    """Activa Per-Monitor DPI v2 antes de capturar o crear el overlay."""
    try:
        if _user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        pass

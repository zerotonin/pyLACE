# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — inspect.navigation                                     ║
# ║  « detection raster + range delimiters + zoomed scrubber »       ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Three stacked horizontal bars sized to the widget width:        ║
# ║                                                                  ║
# ║    1. DetectionRaster — one row per track, dots at every frame   ║
# ║       a detection fired. Colourised per track.                   ║
# ║    2. RangeDelimiter  — two draggable handles ( > and < ) that   ║
# ║       set the zoom window over the full movie span.              ║
# ║    3. ZoomedScrubber  — QSlider whose full width represents only ║
# ║       the delimited subset, so a 90 000-frame video can still be ║
# ║       walked frame-by-frame inside a chosen 500-frame window.    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Three-bar frame-navigation widget for the inspector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from pylace.inspect.traces import TrackTrajectory


@dataclass
class FrameRange:
    """Pure-logic view-range helper, easy to unit-test."""

    total_frames: int
    lo: int = 0
    hi: int = 0

    def __post_init__(self) -> None:
        if self.total_frames < 1:
            raise ValueError("total_frames must be >= 1.")
        if self.hi == 0:
            self.hi = self.total_frames - 1
        self.lo = max(0, min(self.lo, self.total_frames - 1))
        self.hi = max(self.lo + 1, min(self.hi, self.total_frames - 1))

    def fraction(self, frame: int) -> float:
        """Return ``frame`` mapped to ``[0, 1]`` over the *full* movie span."""
        if self.total_frames <= 1:
            return 0.0
        return max(0.0, min(1.0, frame / (self.total_frames - 1)))

    def clamp(self, frame: int) -> int:
        return max(self.lo, min(self.hi, int(frame)))


class _DetectionRaster(QtWidgets.QWidget):
    """One coloured row per track; a dot at every frame the track fired."""

    def __init__(
        self, total_frames: int, parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._range = FrameRange(total_frames=total_frames)
        self._trajectories: list[TrackTrajectory] = []
        self._colours: list[tuple[int, int, int]] = []
        self._current: int = 0
        self.setMinimumHeight(60)

    def set_data(
        self,
        trajectories: list[TrackTrajectory],
        colours: list[tuple[int, int, int]],
    ) -> None:
        self._trajectories = trajectories
        self._colours = colours
        self.update()

    def set_view_range(self, lo: int, hi: int) -> None:
        self._range.lo = max(0, lo)
        self._range.hi = max(self._range.lo + 1, hi)
        self.update()

    def set_current_frame(self, frame: int) -> None:
        self._current = int(frame)
        self.update()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(30, 30, 30))
        w = self.width()
        h = self.height()
        if not self._trajectories or w <= 1:
            return
        n = len(self._trajectories)
        row_h = max(3, (h - 4) // n)
        span = max(1, self._range.hi - self._range.lo)
        for ti, (traj, colour_bgr) in enumerate(
            zip(self._trajectories, self._colours, strict=False),
        ):
            colour = QtGui.QColor(colour_bgr[2], colour_bgr[1], colour_bgr[0])
            painter.setPen(colour)
            mask = (traj.frame_indices >= self._range.lo) & (
                traj.frame_indices <= self._range.hi
            )
            view_frames = traj.frame_indices[mask]
            if view_frames.size == 0:
                continue
            xs = (
                (view_frames - self._range.lo) / span * (w - 1)
            ).astype(np.int32)
            y0 = 2 + ti * row_h
            for x in xs:
                painter.drawLine(int(x), y0, int(x), y0 + row_h - 1)
        if self._range.lo <= self._current <= self._range.hi:
            cx = int((self._current - self._range.lo) / span * (w - 1))
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
            painter.drawLine(cx, 0, cx, h)


class _RangeDelimiter(QtWidgets.QWidget):
    """Two draggable handles (> and <) carving out the zoom window."""

    rangeChanged = QtCore.pyqtSignal(int, int)
    HANDLE_WIDTH_PX = 12

    def __init__(
        self, total_frames: int, parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._total_frames = max(1, total_frames)
        self._lo: int = 0
        self._hi: int = max(0, self._total_frames - 1)
        self._dragging: str | None = None
        self.setMinimumHeight(22)
        self.setMouseTracking(True)

    def lo(self) -> int:
        return self._lo

    def hi(self) -> int:
        return self._hi

    def set_total_frames(self, total: int) -> None:
        self._total_frames = max(1, int(total))
        self._lo = 0
        self._hi = self._total_frames - 1
        self.update()
        self.rangeChanged.emit(self._lo, self._hi)

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(45, 45, 45))
        w = self.width()
        h = self.height()
        if w <= 1:
            return
        x_lo = self._frame_to_x(self._lo)
        x_hi = self._frame_to_x(self._hi)
        painter.fillRect(
            x_lo, 0, max(1, x_hi - x_lo), h, QtGui.QColor(75, 105, 135),
        )
        painter.setPen(QtGui.QPen(QtGui.QColor(240, 215, 50), 2))
        painter.setFont(QtGui.QFont("Sans", 11, QtGui.QFont.Weight.Bold))
        painter.drawText(
            max(0, x_lo - self.HANDLE_WIDTH_PX // 2),
            0, self.HANDLE_WIDTH_PX, h,
            int(Qt.AlignmentFlag.AlignCenter), ">",
        )
        painter.drawText(
            min(w - self.HANDLE_WIDTH_PX, x_hi - self.HANDLE_WIDTH_PX // 2),
            0, self.HANDLE_WIDTH_PX, h,
            int(Qt.AlignmentFlag.AlignCenter), "<",
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        x = int(event.position().x())
        self._dragging = (
            "lo" if abs(x - self._frame_to_x(self._lo))
            <= abs(x - self._frame_to_x(self._hi))
            else "hi"
        )
        self._move_handle_to(x)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._dragging is None:
            return
        self._move_handle_to(int(event.position().x()))

    def mouseReleaseEvent(self, _event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._dragging = None

    def _frame_to_x(self, frame: int) -> int:
        if self._total_frames <= 1:
            return 0
        return int(frame / (self._total_frames - 1) * (self.width() - 1))

    def _x_to_frame(self, x: int) -> int:
        if self.width() <= 1:
            return 0
        f = x / (self.width() - 1) * (self._total_frames - 1)
        return max(0, min(self._total_frames - 1, int(round(f))))

    def _move_handle_to(self, x: int) -> None:
        frame = self._x_to_frame(x)
        if self._dragging == "lo":
            self._lo = min(frame, self._hi - 1)
        elif self._dragging == "hi":
            self._hi = max(frame, self._lo + 1)
        else:
            return
        self.update()
        self.rangeChanged.emit(self._lo, self._hi)


class FrameNavigationStrip(QtWidgets.QWidget):
    """Composite: detection raster + range delimiter + zoomed scrubber."""

    currentFrameChanged = QtCore.pyqtSignal(int)

    def __init__(
        self, total_frames: int, parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._total_frames = max(1, total_frames)

        self._raster = _DetectionRaster(self._total_frames, parent=self)
        self._range_bar = _RangeDelimiter(self._total_frames, parent=self)
        self._scrub = QtWidgets.QSlider(Qt.Orientation.Horizontal, self)
        self._scrub.setRange(0, self._total_frames - 1)
        self._scrub.setMinimumHeight(28)
        self._readout = QtWidgets.QLabel(self)
        self._readout.setMinimumWidth(140)
        self._readout.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        self._range_bar.rangeChanged.connect(self._on_range_changed)
        self._scrub.valueChanged.connect(self._on_scrub_changed)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(2, 2, 2, 2)
        v.setSpacing(2)
        v.addWidget(self._raster)
        v.addWidget(self._range_bar)
        scrub_row = QtWidgets.QHBoxLayout()
        scrub_row.setContentsMargins(0, 0, 0, 0)
        scrub_row.addWidget(self._scrub, stretch=1)
        scrub_row.addWidget(self._readout)
        v.addLayout(scrub_row)

        self._update_readout()

    # ── Public API ─────────────────────────────────────────────────────

    def set_trajectories(
        self,
        trajectories: list[TrackTrajectory],
        colours: list[tuple[int, int, int]],
    ) -> None:
        self._raster.set_data(trajectories, colours)

    def current_frame(self) -> int:
        return int(self._scrub.value())

    def set_current_frame(self, frame: int) -> None:
        clamped = max(self._scrub.minimum(), min(self._scrub.maximum(), int(frame)))
        self._scrub.setValue(clamped)

    def step(self, delta: int) -> None:
        self.set_current_frame(self.current_frame() + int(delta))

    # ── Slots ─────────────────────────────────────────────────────────

    def _on_range_changed(self, lo: int, hi: int) -> None:
        self._scrub.blockSignals(True)
        self._scrub.setRange(lo, hi)
        if self._scrub.value() < lo:
            self._scrub.setValue(lo)
        elif self._scrub.value() > hi:
            self._scrub.setValue(hi)
        self._scrub.blockSignals(False)
        self._raster.set_view_range(lo, hi)
        self._raster.set_current_frame(self._scrub.value())
        self._update_readout()
        self.currentFrameChanged.emit(self._scrub.value())

    def _on_scrub_changed(self, value: int) -> None:
        self._raster.set_current_frame(value)
        self._update_readout()
        self.currentFrameChanged.emit(int(value))

    def _update_readout(self) -> None:
        self._readout.setText(
            f"{self._scrub.value()}  "
            f"({self._range_bar.lo()}–{self._range_bar.hi()})",
        )


__all__ = ["FrameNavigationStrip", "FrameRange"]

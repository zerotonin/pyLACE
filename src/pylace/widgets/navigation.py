# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — widgets.navigation                                     ║
# ║  « detection raster + range delimiters + zoomed scrubber »       ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Three stacked horizontal bars sized to the widget width:        ║
# ║                                                                  ║
# ║    1. _DetectionRaster — one row per track, dots at every        ║
# ║       frame a detection fired, colour-coded; a white cursor     ║
# ║       line marks the current frame.                              ║
# ║    2. _RangeDelimiter  — two draggable handles carving out a     ║
# ║       zoom window over the full movie span; outside the window   ║
# ║       a soft dim wash; double-click anywhere to reset.           ║
# ║    3. _ZoomedScrubber  — a styled QSlider whose full extent      ║
# ║       represents only the delimited subset, so a 90 k-frame      ║
# ║       movie can still be walked frame-by-frame inside a chosen   ║
# ║       500-frame window.                                          ║
# ║                                                                  ║
# ║  Re-used by both the inspector and the post-hoc explorer. All    ║
# ║  visuals share a small PALETTE constant so retheming is one      ║
# ║  edit.                                                           ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Three-bar frame-navigation widget shared across GUIs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from pylace.inspect.traces import TrackTrajectory

# ─────────────────────────────────────────────────────────────────
#  Palette  « single source of truth for the widget's look »
# ─────────────────────────────────────────────────────────────────
PALETTE = {
    "raster_bg":      QtGui.QColor(28, 30, 34),
    "raster_grid":    QtGui.QColor(60, 64, 70),
    "raster_dim_wash": QtGui.QColor(0, 0, 0, 110),  # alpha overlay outside zoom
    "delim_bg":       QtGui.QColor(38, 41, 46),
    "delim_track":    QtGui.QColor(85, 120, 165),       # selected window fill
    "delim_track_glow": QtGui.QColor(115, 160, 220, 60),
    "delim_handle":   QtGui.QColor(245, 200, 60),
    "delim_handle_grip": QtGui.QColor(255, 240, 180),
    "cursor":         QtGui.QColor(255, 255, 255),
    "label":          QtGui.QColor(180, 185, 195),
}

SLIDER_QSS = """
QSlider::groove:horizontal {
    background: #2a2d33;
    height: 6px;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #f2c63c;
    height: 6px;
    border-radius: 3px;
}
QSlider::add-page:horizontal {
    background: #444851;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #fff7d8;
    border: 1px solid #b8932e;
    width: 14px;
    margin: -6px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #ffe28a;
}
"""


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
        """Return ``frame`` mapped to ``[0, 1]`` over the full movie span."""
        if self.total_frames <= 1:
            return 0.0
        return max(0.0, min(1.0, frame / (self.total_frames - 1)))

    def clamp(self, frame: int) -> int:
        return max(self.lo, min(self.hi, int(frame)))


# ─────────────────────────────────────────────────────────────────
#  Detection raster
# ─────────────────────────────────────────────────────────────────


class _DetectionRaster(QtWidgets.QWidget):
    """One coloured row per track; a dot at every frame the track fired.

    The detection dots are painted into a cached QPixmap once per
    resize / data change, and ``paintEvent`` only blits the cache and
    overlays the current-frame cursor. This keeps cursor scrubbing
    cheap even on a 90 k-frame multi-track recording.
    """

    HEIGHT_PX = 64

    def __init__(
        self, total_frames: int, parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._range = FrameRange(total_frames=total_frames)
        self._trajectories: list[TrackTrajectory] = []
        self._colours: list[tuple[int, int, int]] = []
        self._current: int = 0
        self._cache: QtGui.QPixmap | None = None
        self.setMinimumHeight(self.HEIGHT_PX)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def set_data(
        self,
        trajectories: list[TrackTrajectory],
        colours: list[tuple[int, int, int]],
    ) -> None:
        self._trajectories = trajectories
        self._colours = colours
        self._cache = None
        self.update()

    def set_range(self, lo: int, hi: int) -> None:
        if lo == self._range.lo and hi == self._range.hi:
            return
        self._range.lo = lo
        self._range.hi = hi
        self._cache = None
        self.update()

    def set_current_frame(self, frame: int) -> None:
        self._current = int(frame)
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._cache = None

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        if self._cache is None or self._cache.size() != self.size():
            self._cache = self._render_to_pixmap()
        painter.drawPixmap(0, 0, self._cache)
        # Cursor.
        if self._range.lo <= self._current <= self._range.hi and self._cache:
            w = self.width()
            span = max(1, self._range.hi - self._range.lo)
            cx = int((self._current - self._range.lo) / span * (w - 1))
            painter.setPen(QtGui.QPen(PALETTE["cursor"], 1.5))
            painter.drawLine(cx, 0, cx, self.height())

    def _render_to_pixmap(self) -> QtGui.QPixmap:
        pix = QtGui.QPixmap(self.size())
        pix.fill(PALETTE["raster_bg"])
        if self.width() <= 1 or not self._trajectories:
            return pix
        painter = QtGui.QPainter(pix)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        w = self.width()
        h = self.height()

        # Subtle horizontal grid between rows.
        n = len(self._trajectories)
        row_h = max(3, (h - 8) // n)
        painter.setPen(QtGui.QPen(PALETTE["raster_grid"], 1, Qt.PenStyle.DotLine))
        for ti in range(1, n):
            y = 4 + ti * row_h
            painter.drawLine(0, y, w, y)

        # Per-track dots.
        span = max(1, self._range.hi - self._range.lo)
        for ti, (traj, colour_bgr) in enumerate(
            zip(self._trajectories, self._colours, strict=False),
        ):
            colour = QtGui.QColor(colour_bgr[2], colour_bgr[1], colour_bgr[0])
            painter.setPen(QtGui.QPen(colour, 1))
            mask = (traj.frame_indices >= self._range.lo) & (
                traj.frame_indices <= self._range.hi
            )
            view_frames = traj.frame_indices[mask]
            if view_frames.size == 0:
                continue
            xs = (
                (view_frames - self._range.lo) / span * (w - 1)
            ).astype(np.int32)
            y0 = 4 + ti * row_h
            y1 = y0 + row_h - 2
            # Build a list of QLine and draw all at once — much faster
            # than per-detection drawLine calls on a 90 k-frame video.
            lines = [QtCore.QLine(int(x), y0, int(x), y1) for x in xs]
            painter.drawLines(lines)

        # Track index labels along the left edge.
        painter.setPen(PALETTE["label"])
        painter.setFont(QtGui.QFont("Sans", 8))
        for ti, traj in enumerate(self._trajectories):
            y0 = 4 + ti * row_h
            painter.drawText(
                3, y0, 16, row_h - 1,
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                str(traj.track_id),
            )
        painter.end()
        return pix


# ─────────────────────────────────────────────────────────────────
#  Range delimiter
# ─────────────────────────────────────────────────────────────────


class _RangeDelimiter(QtWidgets.QWidget):
    """Two draggable handles carving out a zoom window over the full span.

    The selected window glows in slate-blue with a soft halo; outside
    is drawn at half opacity so the carved-out region is visually
    obvious. Double-clicking anywhere resets the window to the full
    movie span.
    """

    rangeChanged = QtCore.pyqtSignal(int, int)
    HANDLE_HALF_WIDTH = 7  # px from handle centre to its outer edge
    GRAB_RADIUS_PX = 12    # how close to a handle a click counts as a grab
    HEIGHT_PX = 26

    def __init__(
        self, total_frames: int, parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._total_frames = max(1, total_frames)
        self._lo: int = 0
        self._hi: int = max(0, self._total_frames - 1)
        self._dragging: str | None = None
        self.setMinimumHeight(self.HEIGHT_PX)
        self.setMouseTracking(True)
        self.setToolTip(
            "Drag the yellow handles to zoom the scrubber below.\n"
            "Double-click to reset the range to the whole movie.",
        )

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
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        w = self.width()
        h = self.height()

        # Background strip.
        painter.fillRect(self.rect(), PALETTE["delim_bg"])

        x_lo = self._frame_to_x(self._lo)
        x_hi = self._frame_to_x(self._hi)

        # Soft glow behind the selected window — a slightly wider rect
        # behind the main fill at low alpha.
        glow_pad = 3
        painter.fillRect(
            x_lo - glow_pad, glow_pad,
            max(1, x_hi - x_lo + 2 * glow_pad), h - 2 * glow_pad,
            PALETTE["delim_track_glow"],
        )

        # Selected window fill, slightly inset, rounded.
        track_rect = QtCore.QRectF(x_lo, 6, max(1, x_hi - x_lo), h - 12)
        path = QtGui.QPainterPath()
        path.addRoundedRect(track_rect, 4, 4)
        painter.fillPath(path, PALETTE["delim_track"])

        # Handles — slim vertical pill with grip dots.
        for x in (x_lo, x_hi):
            self._paint_handle(painter, x, h)

    def _paint_handle(
        self, painter: QtGui.QPainter, x: int, h: int,
    ) -> None:
        rect = QtCore.QRectF(
            x - self.HANDLE_HALF_WIDTH, 2,
            self.HANDLE_HALF_WIDTH * 2, h - 4,
        )
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, 4, 4)
        painter.fillPath(path, PALETTE["delim_handle"])
        painter.setPen(QtGui.QPen(QtGui.QColor(180, 140, 30), 1))
        painter.drawPath(path)
        # Three grip dots in the centre.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(PALETTE["delim_handle_grip"])
        cx = x
        cy = h // 2
        for dy in (-5, 0, 5):
            painter.drawEllipse(QtCore.QPointF(cx, cy + dy), 1.4, 1.4)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        x = int(event.position().x())
        x_lo = self._frame_to_x(self._lo)
        x_hi = self._frame_to_x(self._hi)
        # Pick the handle within grab radius; otherwise the closer one.
        if abs(x - x_lo) <= self.GRAB_RADIUS_PX and abs(x - x_lo) <= abs(x - x_hi):
            self._dragging = "lo"
        elif abs(x - x_hi) <= self.GRAB_RADIUS_PX:
            self._dragging = "hi"
        else:
            self._dragging = "lo" if abs(x - x_lo) <= abs(x - x_hi) else "hi"
        self._move_handle_to(x)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._dragging is None:
            x = int(event.position().x())
            x_lo = self._frame_to_x(self._lo)
            x_hi = self._frame_to_x(self._hi)
            near = (abs(x - x_lo) <= self.GRAB_RADIUS_PX
                    or abs(x - x_hi) <= self.GRAB_RADIUS_PX)
            self.setCursor(
                Qt.CursorShape.SplitHCursor if near else Qt.CursorShape.ArrowCursor,
            )
            return
        self._move_handle_to(int(event.position().x()))

    def mouseReleaseEvent(self, _event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._dragging = None

    def mouseDoubleClickEvent(self, _event: QtGui.QMouseEvent) -> None:  # noqa: N802
        """Reset the window to span the whole movie."""
        self._lo = 0
        self._hi = max(1, self._total_frames - 1)
        self.update()
        self.rangeChanged.emit(self._lo, self._hi)

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


# ─────────────────────────────────────────────────────────────────
#  Composite strip
# ─────────────────────────────────────────────────────────────────


class FrameNavigationStrip(QtWidgets.QWidget):
    """Composite: detection raster + range delimiter + zoomed scrubber.

    Public API
    ----------
    ``set_trajectories(trajectories, colours)``
        Drive the detection raster.
    ``set_fps(fps)``
        If known, the readout switches from "frame N (range a-b)"
        to "frame N  M.MM s   (range a-b)".
    ``set_current_frame(frame)``
        Programmatic scrub.
    ``current_frame()``
        Current scrubber value.
    ``step(delta)``
        Bump the cursor by ``delta`` frames.
    ``currentFrameChanged``
        Signal emitted when the scrubber changes (by user or
        programmatic call).
    """

    currentFrameChanged = QtCore.pyqtSignal(int)

    def __init__(
        self, total_frames: int, parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._total_frames = max(1, total_frames)
        self._fps: float | None = None

        self._raster = _DetectionRaster(self._total_frames, parent=self)
        self._range_bar = _RangeDelimiter(self._total_frames, parent=self)
        self._scrub = QtWidgets.QSlider(Qt.Orientation.Horizontal, self)
        self._scrub.setRange(0, self._total_frames - 1)
        self._scrub.setMinimumHeight(28)
        self._scrub.setStyleSheet(SLIDER_QSS)
        self._readout = QtWidgets.QLabel(self)
        self._readout.setMinimumWidth(220)
        self._readout.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        font = self._readout.font()
        font.setStyleHint(QtGui.QFont.StyleHint.TypeWriter)
        font.setFamily("Monospace")
        self._readout.setFont(font)

        self._range_bar.rangeChanged.connect(self._on_range_changed)
        self._scrub.valueChanged.connect(self._on_scrub_changed)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(2, 2, 2, 2)
        v.setSpacing(3)
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

    def set_fps(self, fps: float | None) -> None:
        self._fps = float(fps) if fps and fps > 0 else None
        self._update_readout()

    def current_frame(self) -> int:
        return int(self._scrub.value())

    def set_current_frame(self, frame: int) -> None:
        clamped = max(self._scrub.minimum(), min(self._scrub.maximum(), int(frame)))
        self._scrub.setValue(clamped)

    def step(self, delta: int) -> None:
        self.set_current_frame(self.current_frame() + int(delta))

    # ── Slots ─────────────────────────────────────────────────────────

    def _on_range_changed(self, lo: int, hi: int) -> None:
        # Only the bottom scrubber's range follows the delimiters. The
        # raster and the delimiter bar both stay anchored to the full
        # movie span so the user keeps the global view while scrubbing
        # a sub-window.
        self._scrub.blockSignals(True)
        self._scrub.setRange(lo, hi)
        if self._scrub.value() < lo:
            self._scrub.setValue(lo)
        elif self._scrub.value() > hi:
            self._scrub.setValue(hi)
        self._scrub.blockSignals(False)
        self._raster.set_current_frame(self._scrub.value())
        self._update_readout()
        self.currentFrameChanged.emit(self._scrub.value())

    def _on_scrub_changed(self, value: int) -> None:
        self._raster.set_current_frame(value)
        self._update_readout()
        self.currentFrameChanged.emit(int(value))

    def _update_readout(self) -> None:
        f = self._scrub.value()
        lo = self._range_bar.lo()
        hi = self._range_bar.hi()
        if self._fps:
            t_s = f / self._fps
            self._readout.setText(
                f"frame {f:>7d}  {t_s:>8.2f} s   ({lo}–{hi})",
            )
        else:
            self._readout.setText(f"frame {f:>7d}   ({lo}–{hi})")


__all__ = ["FrameNavigationStrip", "FrameRange"]

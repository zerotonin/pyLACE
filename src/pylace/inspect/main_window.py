# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — inspect.main_window                                    ║
# ║  « scrub view + static-trajectory overview, side by side »       ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Left pane: live video frame at the slider's index, with a       ║
# ║  fading trail behind each track and a labelled circle at each    ║
# ║  current detection. Right pane: static background (sample frame  ║
# ║  / max bg / min bg) with every track's full trajectory drawn     ║
# ║  on top, plus a current-frame marker per track.                  ║
# ╚══════════════════════════════════════════════════════════════════╝
"""PyQt6 window for inspecting tracked detections on top of a video."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from pylace.annotator.sidecar import (
    Sidecar,
    SidecarSchemaError,
    default_sidecar_path,
    read_sidecar,
)
from pylace.detect.background import (
    default_background_paths,
    load_background_png,
)
from pylace.widgets.navigation import FrameNavigationStrip
from pylace.inspect.palette import palette_bgr
from pylace.inspect.traces import (
    TrackTrajectory,
    read_traces,
    render_arena_outline,
    render_current_markers,
    render_full_trajectories,
    render_roi_outlines,
    render_trail,
)
from pylace.roi.geometry import ROISet
from pylace.roi.sidecar import (
    ROISidecarSchemaError,
    default_rois_path,
    read_rois,
)

DEFAULT_TRAIL_SECONDS = 5.0


class _ClickableImageLabel(QtWidgets.QLabel):
    """QLabel that emits clicks in source-image (pixel) coordinates.

    The pixmap is scaled with KeepAspectRatio and centred inside the label,
    so we have to undo both the centring offset and the scale to recover
    the original-image coordinate the user actually clicked on.
    """

    imageClicked = QtCore.pyqtSignal(float, float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._image_size: tuple[int, int] | None = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip(
            "Click anywhere on a trajectory to jump the live frame to "
            "that detection.",
        )

    def set_image_size(self, width: int, height: int) -> None:
        self._image_size = (int(width), int(height))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._image_size is None:
            return super().mousePressEvent(event)
        pix = self.pixmap()
        if pix is None or pix.isNull():
            return
        pw, ph = pix.width(), pix.height()
        lw, lh = self.width(), self.height()
        x_off = (lw - pw) / 2.0
        y_off = (lh - ph) / 2.0
        cx = event.position().x() - x_off
        cy = event.position().y() - y_off
        if cx < 0 or cy < 0 or cx > pw or cy > ph:
            return
        sw, sh = self._image_size
        ix = cx * sw / pw
        iy = cy * sh / ph
        self.imageClicked.emit(float(ix), float(iy))


class InspectorWindow(QtWidgets.QMainWindow):
    """Top-level inspector window."""

    def __init__(
        self, video: Path, csv_path: Path, parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._video = video
        self._csv_path = csv_path
        self._trajectories: list[TrackTrajectory] = read_traces(csv_path)
        self._colours = palette_bgr(len(self._trajectories))
        self._cap = cv2.VideoCapture(str(video))
        if not self._cap.isOpened():
            raise OSError(f"Cannot open video: {video}")
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_size = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        self._first_frame = self._read_frame_at(self._first_frame_idx())
        self._bg_max, self._bg_min = self._load_bg_pair()
        self._sidecar = self._load_sidecar()
        self._roi_set = self._load_roi_set()
        self._overview_bg = self._first_frame.copy()
        self._overview_cache: np.ndarray | None = None

        self._current_frame = self._first_frame_idx()
        self._trail_seconds = DEFAULT_TRAIL_SECONDS
        self._show_arena = self._sidecar is not None
        self._show_rois = self._roi_set is not None

        self.setWindowTitle(
            f"pylace-inspect — {video.name}  ({len(self._trajectories)} tracks)",
        )
        self.setCentralWidget(self._build_central())
        self._build_toolbar()
        self.statusBar().showMessage(
            f"Loaded {len(self._trajectories)} tracks across "
            f"{self._total_frames} frames at {self._fps:.1f} fps.",
        )
        self._refresh()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._cap.release()
        super().closeEvent(event)

    # ── Setup ──────────────────────────────────────────────────────────

    def _first_frame_idx(self) -> int:
        if not self._trajectories:
            return 0
        return int(min(t.frame_indices.min() for t in self._trajectories))

    def _read_frame_at(self, idx: int) -> np.ndarray:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, idx)))
        ok, bgr = self._cap.read()
        if not ok:
            return np.zeros(
                (self._frame_size[1], self._frame_size[0], 3), dtype=np.uint8,
            )
        return bgr

    def _load_bg_pair(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        max_path, min_path = default_background_paths(self._video)
        if not (max_path.exists() and min_path.exists()):
            return None, None
        try:
            return load_background_png(max_path), load_background_png(min_path)
        except OSError:
            return None, None

    def _load_sidecar(self) -> Sidecar | None:
        path = default_sidecar_path(self._video)
        if not path.exists():
            return None
        try:
            return read_sidecar(path)
        except (SidecarSchemaError, OSError, ValueError):
            return None

    def _load_roi_set(self) -> ROISet | None:
        path = default_rois_path(self._video)
        if not path.exists():
            return None
        try:
            return read_rois(path).roi_set
        except (ROISidecarSchemaError, OSError, ValueError):
            return None

    # ── Layout ─────────────────────────────────────────────────────────

    def _build_central(self) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget(self)
        v = QtWidgets.QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)

        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal, wrap)
        splitter.addWidget(self._build_scrub_pane())
        splitter.addWidget(self._build_overview_pane())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        v.addWidget(splitter, stretch=1)

        self._nav = FrameNavigationStrip(self._total_frames, parent=wrap)
        self._nav.set_trajectories(self._trajectories, self._colours)
        self._nav.set_fps(self._fps)
        self._nav.set_current_frame(self._current_frame)
        self._nav.currentFrameChanged.connect(self._on_frame_changed)
        v.addWidget(self._nav)
        return wrap

    def _build_scrub_pane(self) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget(self)
        v = QtWidgets.QVBoxLayout(wrap)
        v.setContentsMargins(4, 4, 4, 4)
        v.addWidget(QtWidgets.QLabel("Scrub: live frame + trail", wrap))
        self._scrub_label = QtWidgets.QLabel(wrap)
        self._scrub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scrub_label.setMinimumSize(320, 240)
        v.addWidget(self._scrub_label, stretch=1)
        return wrap

    def _build_overview_pane(self) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget(self)
        v = QtWidgets.QVBoxLayout(wrap)
        v.setContentsMargins(4, 4, 4, 4)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Overview background:", wrap))
        self._cb_bg = QtWidgets.QComboBox(wrap)
        self._cb_bg.addItem("Sample frame (frame 0)", "frame")
        bg_available = self._bg_max is not None and self._bg_min is not None
        self._cb_bg.addItem("Background max", "bg_max")
        self._cb_bg.addItem("Background min", "bg_min")
        if not bg_available:
            for i in (1, 2):
                self._cb_bg.model().item(i).setEnabled(False)
            self._cb_bg.setToolTip(
                "No background sidecars; run pylace-tune or pylace-detect first.",
            )
        self._cb_bg.currentIndexChanged.connect(self._on_bg_changed)
        row.addWidget(self._cb_bg, stretch=1)
        v.addLayout(row)

        self._overview_label = _ClickableImageLabel(wrap)
        self._overview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overview_label.setMinimumSize(320, 240)
        self._overview_label.imageClicked.connect(self._on_overview_clicked)
        v.addWidget(self._overview_label, stretch=1)
        return wrap

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("Controls")
        bar.setMovable(False)

        bar.addWidget(QtWidgets.QLabel(" Trail (s): "))
        self._sb_trail = QtWidgets.QDoubleSpinBox(bar)
        self._sb_trail.setRange(0.0, 120.0)
        self._sb_trail.setSingleStep(0.5)
        self._sb_trail.setValue(self._trail_seconds)
        self._sb_trail.valueChanged.connect(self._on_trail_changed)
        bar.addWidget(self._sb_trail)

        bar.addSeparator()
        self._cb_show_arena = QtWidgets.QCheckBox("Arena", bar)
        self._cb_show_arena.setChecked(self._show_arena)
        self._cb_show_arena.setEnabled(self._sidecar is not None)
        if self._sidecar is None:
            self._cb_show_arena.setToolTip(
                "No arena sidecar found alongside this video.",
            )
        self._cb_show_arena.toggled.connect(self._on_arena_toggled)
        bar.addWidget(self._cb_show_arena)

        self._cb_show_rois = QtWidgets.QCheckBox("ROIs", bar)
        self._cb_show_rois.setChecked(self._show_rois)
        self._cb_show_rois.setEnabled(self._roi_set is not None)
        if self._roi_set is None:
            self._cb_show_rois.setToolTip(
                "No ROI sidecar (.pylace_rois.json) found — pylace-detect "
                "ran on the whole arena. If you expected an ROI, check "
                "that the sidecar lives next to the video.",
            )
        self._cb_show_rois.toggled.connect(self._on_rois_toggled)
        bar.addWidget(self._cb_show_rois)

        bar.addSeparator()
        prev_act = QtGui.QAction("◀ Frame", self)
        prev_act.setShortcut(QtGui.QKeySequence("Left"))
        prev_act.triggered.connect(lambda: self._step_frame(-1))
        bar.addAction(prev_act)
        next_act = QtGui.QAction("Frame ▶", self)
        next_act.setShortcut(QtGui.QKeySequence("Right"))
        next_act.triggered.connect(lambda: self._step_frame(1))
        bar.addAction(next_act)

    # ── Slots ─────────────────────────────────────────────────────────

    def _on_frame_changed(self, value: int) -> None:
        self._current_frame = int(value)
        self._refresh_scrub()
        self._refresh_overview_marker()

    def _on_trail_changed(self, value: float) -> None:
        self._trail_seconds = float(value)
        self._refresh_scrub()

    def _on_bg_changed(self, _index: int) -> None:
        mode = self._cb_bg.currentData() or "frame"
        if mode == "bg_max" and self._bg_max is not None:
            self._overview_bg = cv2.cvtColor(self._bg_max, cv2.COLOR_GRAY2BGR)
        elif mode == "bg_min" and self._bg_min is not None:
            self._overview_bg = cv2.cvtColor(self._bg_min, cv2.COLOR_GRAY2BGR)
        else:
            self._overview_bg = self._first_frame.copy()
        self._overview_cache = None
        self._refresh_overview()

    def _on_arena_toggled(self, checked: bool) -> None:
        self._show_arena = bool(checked)
        self._overview_cache = None
        self._refresh()

    def _on_rois_toggled(self, checked: bool) -> None:
        self._show_rois = bool(checked)
        self._overview_cache = None
        self._refresh()

    def _on_overview_clicked(self, x: float, y: float) -> None:
        """Find the nearest trajectory sample (across all tracks) and jump to it."""
        best_frame: int | None = None
        best_d2 = float("inf")
        best_track: int | None = None
        for traj in self._trajectories:
            if traj.cx_px.size == 0:
                continue
            d2 = (traj.cx_px - x) ** 2 + (traj.cy_px - y) ** 2
            i = int(np.argmin(d2))
            if float(d2[i]) < best_d2:
                best_d2 = float(d2[i])
                best_frame = int(traj.frame_indices[i])
                best_track = int(traj.track_id)
        if best_frame is None:
            return
        self._nav.set_current_frame(best_frame)
        self.statusBar().showMessage(
            f"Jumped to frame {best_frame} on track {best_track} "
            f"(Δ={best_d2 ** 0.5:.1f} px from click)",
            4000,
        )

    def _step_frame(self, delta: int) -> None:
        self._nav.step(delta)

    # ── Render ────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        self._refresh_scrub()
        self._refresh_overview()

    def _refresh_scrub(self) -> None:
        frame = self._read_frame_at(self._current_frame)
        self._draw_geometry(frame)
        trail_frames = max(1, int(round(self._trail_seconds * self._fps)))
        for traj, colour in zip(self._trajectories, self._colours, strict=False):
            render_trail(
                frame, traj, self._current_frame, trail_frames, colour,
            )
        render_current_markers(
            frame, self._trajectories, self._colours, self._current_frame,
        )
        self._set_pixmap(self._scrub_label, frame)

    def _refresh_overview(self) -> None:
        if self._overview_cache is None:
            cache = self._overview_bg.copy()
            self._draw_geometry(cache)
            render_full_trajectories(cache, self._trajectories, self._colours)
            self._overview_cache = cache
        composite = self._overview_cache.copy()
        render_current_markers(
            composite, self._trajectories, self._colours, self._current_frame,
        )
        self._overview_label.set_image_size(
            composite.shape[1], composite.shape[0],
        )
        self._set_pixmap(self._overview_label, composite)

    def _draw_geometry(self, bgr: np.ndarray) -> None:
        """Overlay arena boundary + ROI outlines if those sidecars were loaded."""
        if self._show_arena and self._sidecar is not None:
            render_arena_outline(bgr, self._sidecar.arena)
        if self._show_rois and self._roi_set is not None:
            render_roi_outlines(bgr, self._roi_set)

    def _refresh_overview_marker(self) -> None:
        # Trajectory polylines do not change with the frame; only the
        # current-frame marker does. Re-render fast.
        self._refresh_overview()

    def _set_pixmap(self, label: QtWidgets.QLabel, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        image = QtGui.QImage(
            rgb.tobytes(), w, h, w * 3, QtGui.QImage.Format.Format_RGB888,
        )
        pix = QtGui.QPixmap.fromImage(image)
        label.setPixmap(
            pix.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh()


def run(video: Path, csv_path: Path | None = None) -> int:
    """Launch the inspector. Returns a process exit code."""
    if csv_path is None:
        csv_path = video.with_name(video.name + ".pylace_detections.csv")
    if not csv_path.exists():
        print(
            f"Detections CSV not found: {csv_path}\n"
            "Run pylace-detect on the video first.", file=sys.stderr,
        )
        return 2

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = InspectorWindow(video=video, csv_path=csv_path)
    window.resize(1400, 760)
    window.show()
    return app.exec()


__all__ = ["InspectorWindow", "run"]

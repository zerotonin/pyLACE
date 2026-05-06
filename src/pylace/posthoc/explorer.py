# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — posthoc.explorer                                       ║
# ║  « inspector top half + matplotlib time-series bottom half »     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Top:                                                            ║
# ║    Left  — scrub pane: live current frame, fading trail, arena   ║
# ║            and ROI outlines, current-frame markers.              ║
# ║    Right — overview pane: static background (sample frame /      ║
# ║            max / min projection) with every track's full         ║
# ║            trajectory baked in, plus current-frame markers.      ║
# ║  Bottom:                                                         ║
# ║    Three matplotlib axes sharing the time (s) x-axis: speed,     ║
# ║    yaw rate, distance to wall. Every track is plotted in its     ║
# ║    Wong colour. The matplotlib nav toolbar exposes pan / zoom /  ║
# ║    export so a chosen window can be saved straight to PNG/SVG.   ║
# ║                                                                  ║
# ║  Single source of truth: ``self._current_frame``. Every click —  ║
# ║  on either trajectory pane or any of the three plot axes —       ║
# ║  routes through ``_set_current_frame`` which redraws all four    ║
# ║  cursors (scrub marker, overview marker, three vertical plot     ║
# ║  lines) in lock-step.                                            ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Synchronized data explorer: trajectory panes + matplotlib plots."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from pylace.annotator.sidecar import (
    Sidecar,
    default_sidecar_path,
    read_sidecar,
)
from pylace.detect.background import default_background_paths, load_background_png
from pylace.inspect.main_window import _ClickableImageLabel
from pylace.inspect.palette import palette_bgr
from pylace.inspect.traces import (
    TrackTrajectory,
    render_arena_outline,
    render_current_markers,
    render_full_trajectories,
    render_roi_outlines,
    render_trail,
)
from pylace.posthoc.analytics import compute_distance_to_wall
from pylace.roi.geometry import ROISet
from pylace.roi.sidecar import default_rois_path, read_rois
from pylace.widgets.navigation import FrameNavigationStrip

DEFAULT_TRAIL_SECONDS = 5.0


class ExplorerWindow(QtWidgets.QMainWindow):
    """Synchronized trajectory + kinematic-time-series explorer."""

    def __init__(
        self,
        video: Path,
        trajectory_csv: Path,
        sidecar: Sidecar,
        roi_set: ROISet | None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._video = video
        self._sidecar = sidecar
        self._roi_set = roi_set
        self._fps = float(sidecar.video.fps) or 25.0
        self._pix_per_mm = float(
            sidecar.calibration.pixel_distance / sidecar.calibration.physical_mm,
        )

        self._cap = cv2.VideoCapture(str(video))
        if not self._cap.isOpened():
            raise OSError(f"Cannot open video: {video}")
        self._frame_size = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        df = pd.read_csv(trajectory_csv)
        self._tracks: dict[int, pd.DataFrame] = {
            int(tid): _augment_track(g, sidecar.arena, self._pix_per_mm)
            for tid, g in df.groupby("track_id")
        }
        self._track_ids = sorted(self._tracks.keys())
        self._colours = palette_bgr(len(self._track_ids))
        self._traj_objects = self._build_track_trajectories()

        self._first_frame = self._read_frame_at(self._first_frame_idx())
        self._bg_max, self._bg_min = self._load_bg_pair()
        self._overview_bg = self._first_frame.copy()
        self._overview_cache: np.ndarray | None = None

        self._current_frame = self._first_frame_idx()
        self._trail_seconds = DEFAULT_TRAIL_SECONDS

        self.setWindowTitle(
            f"pylace-explore — {video.name}  "
            f"({len(self._track_ids)} tracks)",
        )
        self.setCentralWidget(self._build_central())
        self._refresh_all()

    # ── Setup helpers ──────────────────────────────────────────────────

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._cap.release()
        super().closeEvent(event)

    def _read_frame_at(self, idx: int) -> np.ndarray:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, idx)))
        ok, bgr = self._cap.read()
        if not ok:
            return np.zeros(
                (self._frame_size[1], self._frame_size[0], 3), dtype=np.uint8,
            )
        return bgr

    def _first_frame_idx(self) -> int:
        if not self._tracks:
            return 0
        return int(min(g["frame_idx"].iloc[0] for g in self._tracks.values()))

    def _build_track_trajectories(self) -> list[TrackTrajectory]:
        out: list[TrackTrajectory] = []
        for tid in self._track_ids:
            g = self._tracks[tid]
            out.append(
                TrackTrajectory(
                    track_id=tid,
                    frame_indices=g["frame_idx"].to_numpy(dtype=np.int64),
                    cx_px=g["cx_smooth_px"].to_numpy(dtype=np.float64),
                    cy_px=g["cy_smooth_px"].to_numpy(dtype=np.float64),
                ),
            )
        return out

    def _load_bg_pair(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        max_path, min_path = default_background_paths(self._video)
        if not (max_path.exists() and min_path.exists()):
            return None, None
        try:
            return load_background_png(max_path), load_background_png(min_path)
        except OSError:
            return None, None

    # ── Layout ─────────────────────────────────────────────────────────

    def _build_central(self) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget(self)
        v = QtWidgets.QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self._build_toolbar())

        splitter_v = QtWidgets.QSplitter(Qt.Orientation.Vertical, wrap)
        splitter_v.addWidget(self._build_top_panes())
        splitter_v.addWidget(self._build_plots_pane())
        splitter_v.setStretchFactor(0, 3)
        splitter_v.setStretchFactor(1, 2)
        v.addWidget(splitter_v, stretch=1)

        self._nav = FrameNavigationStrip(self._total_frames, parent=wrap)
        self._nav.set_trajectories(self._traj_objects, self._colours)
        self._nav.set_fps(self._fps)
        self._nav.set_current_frame(self._current_frame)
        self._nav.currentFrameChanged.connect(self._on_nav_changed)
        v.addWidget(self._nav)
        return wrap

    def _build_toolbar(self) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget(self)
        h = QtWidgets.QHBoxLayout(wrap)
        h.setContentsMargins(4, 4, 4, 4)

        h.addWidget(QtWidgets.QLabel("Trail (s):"))
        self._sb_trail = QtWidgets.QDoubleSpinBox(wrap)
        self._sb_trail.setRange(0.0, 120.0)
        self._sb_trail.setSingleStep(0.5)
        self._sb_trail.setValue(self._trail_seconds)
        self._sb_trail.valueChanged.connect(self._on_trail_changed)
        h.addWidget(self._sb_trail)

        h.addStretch(1)
        self._lbl_status = QtWidgets.QLabel(
            "Click anywhere on the trajectory panes or any plot to scrub time.",
            wrap,
        )
        self._lbl_status.setStyleSheet("color: gray;")
        h.addWidget(self._lbl_status)
        return wrap

    def _build_top_panes(self) -> QtWidgets.QWidget:
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._build_scrub_pane())
        splitter.addWidget(self._build_overview_pane())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        return splitter

    def _build_scrub_pane(self) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget(self)
        v = QtWidgets.QVBoxLayout(wrap)
        v.setContentsMargins(2, 2, 2, 2)
        v.addWidget(QtWidgets.QLabel("Scrub: live frame + trail", wrap))
        self._scrub_label = _ClickableImageLabel(wrap)
        self._scrub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scrub_label.setMinimumSize(320, 240)
        self._scrub_label.setToolTip(
            "Click to jump the time cursor to the nearest trajectory sample.",
        )
        self._scrub_label.imageClicked.connect(self._on_pane_clicked)
        v.addWidget(self._scrub_label, stretch=1)
        return wrap

    def _build_overview_pane(self) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget(self)
        v = QtWidgets.QVBoxLayout(wrap)
        v.setContentsMargins(2, 2, 2, 2)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Overview background:", wrap))
        self._cb_bg = QtWidgets.QComboBox(wrap)
        self._cb_bg.addItem("Sample frame", "frame")
        bg_available = self._bg_max is not None and self._bg_min is not None
        self._cb_bg.addItem("Background max", "bg_max")
        self._cb_bg.addItem("Background min", "bg_min")
        if not bg_available:
            for i in (1, 2):
                self._cb_bg.model().item(i).setEnabled(False)
            self._cb_bg.setToolTip(
                "No background sidecars next to the video; "
                "run pylace-tune or pylace-detect to generate them.",
            )
        self._cb_bg.currentIndexChanged.connect(self._on_bg_changed)
        row.addWidget(self._cb_bg, stretch=1)
        v.addLayout(row)

        self._overview_label = _ClickableImageLabel(wrap)
        self._overview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overview_label.setMinimumSize(320, 240)
        self._overview_label.setToolTip(
            "Click anywhere on a trajectory to jump there.",
        )
        self._overview_label.imageClicked.connect(self._on_pane_clicked)
        v.addWidget(self._overview_label, stretch=1)
        return wrap

    def _build_plots_pane(self) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget(self)
        v = QtWidgets.QVBoxLayout(wrap)
        v.setContentsMargins(2, 2, 2, 2)

        self._fig = Figure(figsize=(8, 4), tight_layout=True)
        self._ax_speed = self._fig.add_subplot(3, 1, 1)
        self._ax_yaw = self._fig.add_subplot(3, 1, 2, sharex=self._ax_speed)
        self._ax_wall = self._fig.add_subplot(3, 1, 3, sharex=self._ax_speed)
        self._ax_speed.set_ylabel("speed (mm/s)")
        self._ax_yaw.set_ylabel("yaw rate (°/s)")
        self._ax_wall.set_ylabel("dist. to wall (mm)")
        self._ax_wall.set_xlabel("time (s)")
        for ax in (self._ax_speed, self._ax_yaw):
            ax.tick_params(axis="x", labelbottom=False)
        for ax in (self._ax_speed, self._ax_yaw, self._ax_wall):
            ax.grid(True, alpha=0.3)

        self._cursor_lines = []
        for ax in (self._ax_speed, self._ax_yaw, self._ax_wall):
            line = ax.axvline(
                x=0, color="#FFB000", linewidth=1.0, zorder=10,
            )
            self._cursor_lines.append(line)

        # Plot every track once. Wong palette → matplotlib RGB.
        for tid, colour_bgr in zip(self._track_ids, self._colours, strict=True):
            track = self._tracks[tid]
            t = track["frame_idx"].to_numpy() / self._fps
            rgb = (
                colour_bgr[2] / 255.0,
                colour_bgr[1] / 255.0,
                colour_bgr[0] / 255.0,
            )
            label = f"track {tid}"
            self._ax_speed.plot(
                t, track["speed_mm_s"].to_numpy(),
                color=rgb, linewidth=0.7, label=label,
            )
            self._ax_yaw.plot(
                t, track["yaw_rate_deg_s"].to_numpy(),
                color=rgb, linewidth=0.7,
            )
            self._ax_wall.plot(
                t, track["distance_to_wall_mm"].to_numpy(),
                color=rgb, linewidth=0.7,
            )
        self._ax_speed.legend(loc="upper right", fontsize=8, framealpha=0.85)

        self._canvas = FigureCanvas(self._fig)
        self._canvas.mpl_connect("button_press_event", self._on_plot_clicked)
        self._mpl_toolbar = NavigationToolbar(self._canvas, wrap)
        v.addWidget(self._mpl_toolbar)
        v.addWidget(self._canvas, stretch=1)
        return wrap


    # ── Slots ─────────────────────────────────────────────────────────

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

    def _on_nav_changed(self, value: int) -> None:
        self._set_current_frame(int(value), source="nav")

    def _on_pane_clicked(self, x: float, y: float) -> None:
        best_frame: int | None = None
        best_d2 = float("inf")
        for traj in self._traj_objects:
            cx = traj.cx_px
            cy = traj.cy_px
            valid = ~(np.isnan(cx) | np.isnan(cy))
            if not valid.any():
                continue
            d2 = np.where(valid, (cx - x) ** 2 + (cy - y) ** 2, np.inf)
            i = int(np.argmin(d2))
            if float(d2[i]) < best_d2:
                best_d2 = float(d2[i])
                best_frame = int(traj.frame_indices[i])
        if best_frame is None:
            return
        self._set_current_frame(best_frame, source="trajectory")

    def _on_plot_clicked(self, event) -> None:
        if event.button != 1 or event.inaxes is None or event.xdata is None:
            return
        frame = int(round(float(event.xdata) * self._fps))
        self._set_current_frame(frame, source="plot")

    # ── State + render ────────────────────────────────────────────────

    def _set_current_frame(self, frame: int, *, source: str) -> None:
        frame = max(0, min(self._total_frames - 1, int(frame)))
        self._current_frame = frame
        if source != "nav":
            self._nav.blockSignals(True)
            self._nav.set_current_frame(frame)
            self._nav.blockSignals(False)
        t_s = frame / self._fps
        for line in self._cursor_lines:
            line.set_xdata([t_s, t_s])
        self._canvas.draw_idle()
        self._refresh_scrub()
        self._refresh_overview_marker()

    def _refresh_all(self) -> None:
        self._refresh_scrub()
        self._refresh_overview()
        t_s = self._current_frame / self._fps
        for line in self._cursor_lines:
            line.set_xdata([t_s, t_s])
        self._canvas.draw_idle()

    def _refresh_scrub(self) -> None:
        frame = self._read_frame_at(self._current_frame).copy()
        render_arena_outline(frame, self._sidecar.arena)
        if self._roi_set is not None:
            render_roi_outlines(frame, self._roi_set)
        trail_frames = max(1, int(round(self._trail_seconds * self._fps)))
        for traj, colour in zip(self._traj_objects, self._colours, strict=False):
            render_trail(
                frame, traj, self._current_frame, trail_frames, colour,
            )
        render_current_markers(
            frame, self._traj_objects, self._colours, self._current_frame,
        )
        self._scrub_label.set_image_size(frame.shape[1], frame.shape[0])
        self._set_pixmap(self._scrub_label, frame)

    def _refresh_overview(self) -> None:
        if self._overview_cache is None:
            cache = self._overview_bg.copy()
            render_arena_outline(cache, self._sidecar.arena)
            if self._roi_set is not None:
                render_roi_outlines(cache, self._roi_set)
            render_full_trajectories(cache, self._traj_objects, self._colours)
            self._overview_cache = cache
        composite = self._overview_cache.copy()
        render_current_markers(
            composite, self._traj_objects, self._colours, self._current_frame,
        )
        self._overview_label.set_image_size(
            composite.shape[1], composite.shape[0],
        )
        self._set_pixmap(self._overview_label, composite)

    def _refresh_overview_marker(self) -> None:
        # Trajectories don't change with frame; only the marker does.
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
            ),
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_scrub()
        self._refresh_overview()


def _augment_track(
    group: pd.DataFrame, arena, pix_per_mm: float,
) -> pd.DataFrame:
    """Add ``distance_to_wall_mm`` to a per-track DataFrame."""
    g = group.sort_values("frame_idx").reset_index(drop=True).copy()
    g["distance_to_wall_mm"] = compute_distance_to_wall(
        g, arena, pix_per_mm=pix_per_mm,
    )
    return g


def run(
    video: Path,
    trajectory: Path,
    sidecar_path: Path | None = None,
) -> int:
    """Launch the explorer GUI; returns a process exit code."""
    sidecar_path = sidecar_path or default_sidecar_path(video)
    if not sidecar_path.exists():
        print(
            f"Arena sidecar not found: {sidecar_path}\n"
            "Run pylace-annotate on this video first.",
            file=sys.stderr,
        )
        return 2
    sidecar = read_sidecar(sidecar_path)

    rois_path = default_rois_path(video)
    roi_set = None
    if rois_path.exists():
        try:
            roi_set = read_rois(rois_path).roi_set
        except Exception:
            roi_set = None

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = ExplorerWindow(
        video=video, trajectory_csv=trajectory,
        sidecar=sidecar, roi_set=roi_set,
    )
    window.resize(1500, 1000)
    window.show()
    return app.exec()


__all__ = ["ExplorerWindow", "run"]

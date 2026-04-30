"""PyQt6 main window for ``pylace-tune``."""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from pylace.annotator.sidecar import (
    Sidecar,
    default_sidecar_path,
    read_sidecar,
    video_sha256,
)
from pylace.detect.arena_mask import arena_mask
from pylace.detect.background import (
    compute_projection_pair,
    default_background_paths,
    detection_and_trail,
    load_background_png,
    save_background_png,
)
from pylace.detect.frame import Detection, detect_blobs_with_mask
from pylace.tune.frame_loader import sample_preview_frames
from pylace.tune.overlay import render_overlay
from pylace.tune.params import (
    BackgroundParams,
    DetectionParams,
    TrackingParams,
    TuningParams,
    TuningParamsSchemaError,
    default_params_path,
    read_params,
    write_params,
)

PREVIEW_DEFAULT_N = 12


class TuneWindow(QtWidgets.QMainWindow):
    """QMainWindow for live tuning of detector + background + sample-window params."""

    def __init__(
        self,
        video: Path,
        sidecar: Sidecar,
        params_path: Path,
    ) -> None:
        super().__init__()
        self._video = video
        self._sidecar = sidecar
        self._params_path = params_path
        self._params = self._load_or_default_params()

        self._mask = arena_mask(sidecar.arena, sidecar.video.frame_size)
        self._background: np.ndarray | None = None
        self._trail: np.ndarray | None = None
        self._bg_max: np.ndarray | None = None
        self._bg_min: np.ndarray | None = None
        self._frames: list[np.ndarray] = []
        self._frame_indices: list[int] = []
        self._current = 0

        self._video_total_frames = self._probe_total_frames()
        fps = sidecar.video.fps if sidecar.video.fps > 0 else 25.0
        self._fps = fps
        self._video_duration_s = self._video_total_frames / fps
        self._preview_start_s = 0.0
        self._preview_end_s = self._video_duration_s
        self._preview_n = PREVIEW_DEFAULT_N

        self._show_mask = False
        self._show_arena = True
        self._show_contours = False
        self._show_ellipses = True
        self._show_centroids = True
        self._view_mode: str = "frame"  # "frame" | "detection_bg" | "trail_bg"

        self.setWindowTitle(f"pylace-tune — {video.name}")
        self._build_ui()
        self._initial_load()

    # ── Loading helpers ────────────────────────────────────────────────

    def _load_or_default_params(self) -> TuningParams:
        if not self._params_path.exists():
            return TuningParams.defaults()
        try:
            params, _ = read_params(self._params_path)
            return params
        except TuningParamsSchemaError:
            return TuningParams.defaults()

    def _probe_total_frames(self) -> int:
        import cv2

        cap = cv2.VideoCapture(str(self._video))
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {self._video}")
        try:
            return max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        finally:
            cap.release()

    def _initial_load(self) -> None:
        self._sample_frames()
        self._load_or_compute_background()
        self._render_current()

    def _load_or_compute_background(self) -> None:
        """Prefer the saved max+min sidecars over a fresh recompute at startup."""
        max_path, min_path = default_background_paths(self._video)
        if max_path.exists() and min_path.exists():
            try:
                self._bg_max = load_background_png(max_path)
                self._bg_min = load_background_png(min_path)
                self._apply_polarity()
                self.statusBar().showMessage(
                    f"Loaded background pair from {max_path.name} / {min_path.name}",
                    4000,
                )
                return
            except OSError:
                pass
        self._rebuild_background()

    # ── Layout ─────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        self.setCentralWidget(splitter)

        self.statusBar().showMessage("Loading...")

    def _build_left_panel(self) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget(self)
        v = QtWidgets.QVBoxLayout(wrap)
        v.setContentsMargins(4, 4, 4, 4)

        self._frame_label = QtWidgets.QLabel(wrap)
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setMinimumSize(200, 200)
        v.addWidget(self._frame_label, stretch=1)

        nav = QtWidgets.QHBoxLayout()
        nav.addWidget(QtWidgets.QLabel("Frame"))
        self._frame_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal, wrap)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(0)
        self._frame_slider.valueChanged.connect(self._on_frame_changed)
        nav.addWidget(self._frame_slider, stretch=1)
        self._frame_index_label = QtWidgets.QLabel("0 / 0", wrap)
        nav.addWidget(self._frame_index_label)
        v.addLayout(nav)

        return wrap

    def _build_right_panel(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(320)

        wrap = QtWidgets.QWidget(scroll)
        v = QtWidgets.QVBoxLayout(wrap)
        v.addWidget(self._build_detection_group(wrap))
        v.addWidget(self._build_background_group(wrap))
        v.addWidget(self._build_tracking_group(wrap))
        v.addWidget(self._build_preview_group(wrap))
        v.addWidget(self._build_display_group(wrap))
        v.addWidget(self._build_stats_group(wrap))
        v.addStretch(1)

        button_row = QtWidgets.QHBoxLayout()
        save = QtWidgets.QPushButton("Save", wrap)
        save.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        save.setToolTip("Save params to <video>.pylace_detect_params.json (Ctrl+S)")
        save.clicked.connect(self._on_save_params)
        button_row.addWidget(save)

        save_as = QtWidgets.QPushButton("Save as…", wrap)
        save_as.setToolTip("Save params to a chosen file as a named preset")
        save_as.clicked.connect(self._on_save_preset_as)
        button_row.addWidget(save_as)

        load = QtWidgets.QPushButton("Load preset…", wrap)
        load.setToolTip("Load a previously-saved preset JSON")
        load.clicked.connect(self._on_load_preset)
        button_row.addWidget(load)

        v.addLayout(button_row)

        edit_bg = QtWidgets.QPushButton("Edit background…", wrap)
        edit_bg.setToolTip(
            "Paint a mask and smart-fill (cv2.inpaint) — useful for removing "
            "a stationary fly that the bg model has baked in.",
        )
        edit_bg.clicked.connect(self._on_edit_background)
        v.addWidget(edit_bg)

        scroll.setWidget(wrap)
        return scroll

    def _build_detection_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Detection", parent)
        form = QtWidgets.QFormLayout(box)
        dp = self._params.detection
        self._sb_threshold = self._spin(form, "Threshold", dp.threshold, 0, 255)
        self._sb_min_area = self._spin(form, "Min area (px)", dp.min_area, 1, 1_000_000)
        self._sb_max_area = self._spin(form, "Max area (px)", dp.max_area, 1, 1_000_000)
        self._sb_morph = self._spin(form, "Morph kernel", dp.morph_kernel, 0, 21)
        self._sb_dilate = self._spin(form, "Dilate iters", dp.dilate_iters, 0, 20)
        self._sb_erode = self._spin(form, "Erode iters", dp.erode_iters, 0, 20)
        self._sb_min_sol = self._dspin(
            form, "Min solidity (0=off)", dp.min_solidity, 0.0, 1.0, 0.01,
        )
        self._sb_min_sol.setToolTip(
            "0 disables. ~0.85 rejects diffuse shadows; lower for side-lying "
            "flies whose silhouette is less convex.",
        )
        self._sb_max_axis = self._dspin(
            form, "Max axis ratio (0=off)", dp.max_axis_ratio, 0.0, 30.0, 0.5,
        )
        self._sb_max_axis.setToolTip(
            "0 disables. ~5.0 rejects long thin streaks (wing reflections).",
        )
        for sb in (
            self._sb_threshold, self._sb_min_area, self._sb_max_area,
            self._sb_morph, self._sb_dilate, self._sb_erode,
        ):
            sb.valueChanged.connect(self._on_detection_changed)
        for dsb in (self._sb_min_sol, self._sb_max_axis):
            dsb.valueChanged.connect(self._on_detection_changed)
        return box

    def _build_background_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Background (projection pair)", parent)
        form = QtWidgets.QFormLayout(box)
        bp = self._params.background
        self._sb_bg_n = self._spin(form, "n_frames", bp.n_frames, 1, 1000)
        self._sb_bg_start = self._dspin(form, "start_frac", bp.start_frac, 0.0, 1.0, 0.01)
        self._sb_bg_end = self._dspin(form, "end_frac", bp.end_frac, 0.0, 1.0, 0.01)

        self._cb_polarity = QtWidgets.QComboBox(box)
        self._cb_polarity.addItem("Dark on light (max → detection)", "dark_on_light")
        self._cb_polarity.addItem("Light on dark (min → detection)", "light_on_dark")
        idx = 0 if bp.polarity == "dark_on_light" else 1
        self._cb_polarity.setCurrentIndex(idx)
        self._cb_polarity.currentIndexChanged.connect(self._on_polarity_changed)
        form.addRow("Polarity", self._cb_polarity)

        rebuild = QtWidgets.QPushButton("Rebuild background", box)
        rebuild.clicked.connect(self._on_rebuild_background)
        form.addRow(rebuild)
        return box

    def _build_tracking_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Tracking + chain split (saved → pylace-detect)", parent)
        form = QtWidgets.QFormLayout(box)
        tp = self._params.tracking

        self._cb_track_enabled = QtWidgets.QCheckBox("Enable Hungarian tracker")
        self._cb_track_enabled.setChecked(tp.enabled)
        form.addRow(self._cb_track_enabled)

        self._sb_n_animals = self._spin(
            form, "N animals (0 = auto)", tp.n_animals or 0, 0, 100,
        )
        self._sb_n_animals.setToolTip(
            "Set to the known animal count for fixed-N mode (LACE-paper "
            "assumption). 0 lets tracks be born and retired automatically.",
        )
        self._sb_track_dist = self._dspin(
            form, "Max distance (px)", tp.max_distance_px, 0.0, 5000.0, 5.0,
        )
        self._sb_track_missed = self._spin(
            form, "Max missed frames", tp.max_missed_frames, 0, 1000,
        )
        self._sb_expected_area = self._dspin(
            form, "Expected animal area px (0 = auto)",
            tp.expected_animal_area_px or 0.0, 0.0, 1_000_000.0, 50.0,
        )
        self._sb_expected_area.setToolTip(
            "Chain splitter cuts a blob whose area exceeds 1.5× this value. "
            "0 = auto-learn the median from the first frames at runtime.",
        )
        self._sb_area_w = self._dspin(
            form, "Area cost weight", tp.area_cost_weight, 0.0, 10.0, 0.01,
        )
        self._sb_area_w.setToolTip(
            "Adds |Δarea|·w to the Hungarian cost. Helps disambiguate two "
            "animals at similar positions but different sizes.",
        )
        self._sb_per_w = self._dspin(
            form, "Perimeter cost weight", tp.perimeter_cost_weight, 0.0, 10.0, 0.01,
        )
        self._sb_per_w.setToolTip(
            "Adds |Δperimeter|·w to the Hungarian cost.",
        )

        self._cb_track_enabled.toggled.connect(self._on_tracking_changed)
        for sb in (self._sb_n_animals, self._sb_track_missed):
            sb.valueChanged.connect(self._on_tracking_changed)
        for dsb in (
            self._sb_track_dist, self._sb_expected_area,
            self._sb_area_w, self._sb_per_w,
        ):
            dsb.valueChanged.connect(self._on_tracking_changed)
        return box

    def _build_preview_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Preview window", parent)
        form = QtWidgets.QFormLayout(box)
        self._sb_prev_start = self._dspin(
            form, "Start (s)", self._preview_start_s, 0.0, self._video_duration_s, 1.0,
        )
        self._sb_prev_end = self._dspin(
            form, "End (s)", self._preview_end_s, 0.0, self._video_duration_s, 1.0,
        )
        self._sb_prev_n = self._spin(form, "Sample N frames", self._preview_n, 1, 200)
        resample = QtWidgets.QPushButton("Resample", box)
        resample.clicked.connect(self._on_resample)
        form.addRow(resample)
        return box

    def _build_display_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Display", parent)
        v = QtWidgets.QVBoxLayout(box)

        view_row = QtWidgets.QHBoxLayout()
        view_row.addWidget(QtWidgets.QLabel("Show:", box))
        self._cb_view = QtWidgets.QComboBox(box)
        self._cb_view.addItem("Sample frame", "frame")
        self._cb_view.addItem("Detection bg", "detection_bg")
        self._cb_view.addItem("Trail bg (animal heatmap)", "trail_bg")
        self._cb_view.currentIndexChanged.connect(self._on_view_mode_changed)
        view_row.addWidget(self._cb_view, stretch=1)
        v.addLayout(view_row)

        self._cb_mask = self._check(v, "Foreground mask tint", self._show_mask)
        self._cb_arena = self._check(v, "Arena outline", self._show_arena)
        self._cb_contours = self._check(v, "Contours", self._show_contours)
        self._cb_ellipses = self._check(v, "Ellipses", self._show_ellipses)
        self._cb_centroids = self._check(v, "Centroids", self._show_centroids)
        for cb in (
            self._cb_mask, self._cb_arena, self._cb_contours,
            self._cb_ellipses, self._cb_centroids,
        ):
            cb.toggled.connect(self._on_display_toggled)
        return box

    def _build_stats_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Stats", parent)
        v = QtWidgets.QVBoxLayout(box)
        self._lbl_frame = QtWidgets.QLabel("Frame: —", box)
        self._lbl_this = QtWidgets.QLabel("This frame: — detections", box)
        self._lbl_agg = QtWidgets.QLabel("Across sample: — / —", box)
        self._lbl_shape = QtWidgets.QLabel("Shape: —", box)
        v.addWidget(self._lbl_frame)
        v.addWidget(self._lbl_this)
        v.addWidget(self._lbl_agg)
        v.addWidget(self._lbl_shape)
        return box

    # ── Widget builders ────────────────────────────────────────────────

    def _spin(
        self, form: QtWidgets.QFormLayout, label: str, value: int, lo: int, hi: int,
    ) -> QtWidgets.QSpinBox:
        sb = QtWidgets.QSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(int(value))
        form.addRow(label, sb)
        return sb

    def _dspin(
        self, form: QtWidgets.QFormLayout, label: str,
        value: float, lo: float, hi: float, step: float,
    ) -> QtWidgets.QDoubleSpinBox:
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setSingleStep(step)
        sb.setDecimals(3)
        sb.setValue(float(value))
        form.addRow(label, sb)
        return sb

    def _check(
        self, layout: QtWidgets.QVBoxLayout, label: str, value: bool,
    ) -> QtWidgets.QCheckBox:
        cb = QtWidgets.QCheckBox(label)
        cb.setChecked(value)
        layout.addWidget(cb)
        return cb

    # ── Event handlers ─────────────────────────────────────────────────

    def _on_detection_changed(self) -> None:
        self._params = TuningParams(
            detection=DetectionParams(
                threshold=self._sb_threshold.value(),
                min_area=self._sb_min_area.value(),
                max_area=self._sb_max_area.value(),
                morph_kernel=self._sb_morph.value(),
                dilate_iters=self._sb_dilate.value(),
                erode_iters=self._sb_erode.value(),
                min_solidity=float(self._sb_min_sol.value()),
                max_axis_ratio=float(self._sb_max_axis.value()),
            ),
            background=self._params.background,
            tracking=self._params.tracking,
        )
        self._render_current()

    def _on_tracking_changed(self) -> None:
        n = self._sb_n_animals.value()
        area = float(self._sb_expected_area.value())
        self._params = TuningParams(
            detection=self._params.detection,
            background=self._params.background,
            tracking=TrackingParams(
                enabled=self._cb_track_enabled.isChecked(),
                max_distance_px=float(self._sb_track_dist.value()),
                max_missed_frames=self._sb_track_missed.value(),
                n_animals=n if n > 0 else None,
                expected_animal_area_px=area if area > 0 else None,
                area_cost_weight=float(self._sb_area_w.value()),
                perimeter_cost_weight=float(self._sb_per_w.value()),
            ),
        )

    def _on_display_toggled(self) -> None:
        self._show_mask = self._cb_mask.isChecked()
        self._show_arena = self._cb_arena.isChecked()
        self._show_contours = self._cb_contours.isChecked()
        self._show_ellipses = self._cb_ellipses.isChecked()
        self._show_centroids = self._cb_centroids.isChecked()
        self._render_current()

    def _on_view_mode_changed(self, _index: int) -> None:
        self._view_mode = self._cb_view.currentData() or "frame"
        self._render_current()

    def _on_polarity_changed(self, _index: int) -> None:
        new_polarity = self._cb_polarity.currentData() or "dark_on_light"
        self._params = TuningParams(
            detection=self._params.detection,
            background=BackgroundParams(
                n_frames=self._params.background.n_frames,
                start_frac=self._params.background.start_frac,
                end_frac=self._params.background.end_frac,
                polarity=new_polarity,
            ),
            tracking=self._params.tracking,
        )
        self._apply_polarity()
        self._render_current()

    def _on_rebuild_background(self) -> None:
        self._params = TuningParams(
            detection=self._params.detection,
            background=BackgroundParams(
                n_frames=self._sb_bg_n.value(),
                start_frac=float(self._sb_bg_start.value()),
                end_frac=float(self._sb_bg_end.value()),
                polarity=self._params.background.polarity,
            ),
            tracking=self._params.tracking,
        )
        self.statusBar().showMessage("Rebuilding background...")
        QtWidgets.QApplication.processEvents()
        self._rebuild_background()
        self._render_current()
        self.statusBar().showMessage("Background rebuilt.", 3000)

    def _on_resample(self) -> None:
        self._preview_start_s = float(self._sb_prev_start.value())
        self._preview_end_s = float(self._sb_prev_end.value())
        self._preview_n = self._sb_prev_n.value()
        if self._preview_end_s <= self._preview_start_s:
            self.statusBar().showMessage("End time must be after start.", 3000)
            return
        self.statusBar().showMessage("Resampling preview frames...")
        QtWidgets.QApplication.processEvents()
        self._sample_frames()
        self._render_current()
        self.statusBar().showMessage(
            f"Sampled {len(self._frames)} frames.", 3000,
        )

    def _on_frame_changed(self, idx: int) -> None:
        if not self._frames:
            return
        self._current = max(0, min(len(self._frames) - 1, int(idx)))
        self._render_current()

    def _on_save_params(self) -> None:
        write_params(
            self._params,
            video_path=self._video,
            video_sha256_hex=self._sidecar.video.sha256,
            out_path=self._params_path,
        )
        self.statusBar().showMessage(
            f"Saved params to {self._params_path.name}", 5000,
        )

    def _on_edit_background(self) -> None:
        if self._background is None:
            self.statusBar().showMessage("No background loaded yet.", 3000)
            return
        from pylace.tune.bg_editor import BgEditDialog

        dlg = BgEditDialog(self._background, parent=self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        edited = dlg.result_bg()
        self._background = edited
        # The dialog edits the *detection* bg, which is whichever projection
        # the current polarity routes to. Save back to that projection's
        # sidecar so the next pylace-detect run picks it up.
        max_path, min_path = default_background_paths(self._video)
        target_path = (
            max_path
            if self._params.background.polarity == "dark_on_light"
            else min_path
        )
        if self._params.background.polarity == "dark_on_light":
            self._bg_max = edited
        else:
            self._bg_min = edited
        try:
            save_background_png(edited, target_path)
        except OSError as exc:
            self.statusBar().showMessage(
                f"Background edited but could not save: {exc}", 6000,
            )
        else:
            self.statusBar().showMessage(
                f"Edited bg saved to {target_path.name}.", 4000,
            )
        self._render_current()

    def _on_save_preset_as(self) -> None:
        path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save preset",
            str(self._params_path.parent / "preset.pylace_detect_params.json"),
            "pyLACE preset (*.json);;All files (*)",
        )
        if not path_str:
            return
        target = Path(path_str)
        write_params(
            self._params,
            video_path=self._video,
            video_sha256_hex=self._sidecar.video.sha256,
            out_path=target,
        )
        self.statusBar().showMessage(f"Saved preset to {target.name}", 5000)

    def _on_load_preset(self) -> None:
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load preset",
            str(self._params_path.parent),
            "pyLACE preset (*.json);;All files (*)",
        )
        if not path_str:
            return
        try:
            params, _ = read_params(Path(path_str))
        except (TuningParamsSchemaError, ValueError, KeyError, OSError) as exc:
            self.statusBar().showMessage(f"Cannot load preset: {exc}", 7000)
            return
        self._params = params
        self._sync_widgets_from_params()
        self._rebuild_background()
        self._render_current()
        self.statusBar().showMessage(f"Loaded preset {Path(path_str).name}", 5000)

    def _sync_widgets_from_params(self) -> None:
        """Push self._params back into the spinboxes without triggering signals."""
        dp = self._params.detection
        bp = self._params.background
        tp = self._params.tracking
        for sb, val in (
            (self._sb_threshold, dp.threshold),
            (self._sb_min_area, dp.min_area),
            (self._sb_max_area, dp.max_area),
            (self._sb_morph, dp.morph_kernel),
            (self._sb_dilate, dp.dilate_iters),
            (self._sb_erode, dp.erode_iters),
            (self._sb_bg_n, bp.n_frames),
            (self._sb_n_animals, tp.n_animals or 0),
            (self._sb_track_missed, tp.max_missed_frames),
        ):
            sb.blockSignals(True)
            sb.setValue(int(val))
            sb.blockSignals(False)
        for sb, val in (
            (self._sb_bg_start, bp.start_frac),
            (self._sb_bg_end, bp.end_frac),
            (self._sb_min_sol, dp.min_solidity),
            (self._sb_max_axis, dp.max_axis_ratio),
            (self._sb_track_dist, tp.max_distance_px),
            (self._sb_expected_area, tp.expected_animal_area_px or 0.0),
            (self._sb_area_w, tp.area_cost_weight),
            (self._sb_per_w, tp.perimeter_cost_weight),
        ):
            sb.blockSignals(True)
            sb.setValue(float(val))
            sb.blockSignals(False)
        self._cb_track_enabled.blockSignals(True)
        self._cb_track_enabled.setChecked(tp.enabled)
        self._cb_track_enabled.blockSignals(False)
        self._cb_polarity.blockSignals(True)
        self._cb_polarity.setCurrentIndex(
            0 if bp.polarity == "dark_on_light" else 1,
        )
        self._cb_polarity.blockSignals(False)

    # ── Compute / render ───────────────────────────────────────────────

    def _sample_frames(self) -> None:
        start_frame = int(round(self._preview_start_s * self._fps))
        end_frame = int(round(self._preview_end_s * self._fps))
        end_frame = min(end_frame, self._video_total_frames)
        end_frame = max(end_frame, start_frame + 1)
        self._frames, self._frame_indices = sample_preview_frames(
            self._video, n=self._preview_n,
            start_frame=start_frame, end_frame=end_frame,
        )
        self._current = 0
        self._frame_slider.blockSignals(True)
        self._frame_slider.setMaximum(max(0, len(self._frames) - 1))
        self._frame_slider.setValue(0)
        self._frame_slider.blockSignals(False)

    def _rebuild_background(self) -> None:
        bp = self._params.background
        self._bg_max, self._bg_min = compute_projection_pair(
            self._video, n_frames=bp.n_frames,
            start_frac=bp.start_frac, end_frac=bp.end_frac,
        )
        max_path, min_path = default_background_paths(self._video)
        try:
            save_background_png(self._bg_max, max_path)
            save_background_png(self._bg_min, min_path)
        except OSError as exc:
            self.statusBar().showMessage(f"Could not save background: {exc}", 5000)
        self._apply_polarity()

    def _apply_polarity(self) -> None:
        """Route the cached max/min projections into detection_bg / trail_bg."""
        if self._bg_max is None or self._bg_min is None:
            return
        self._background, self._trail = detection_and_trail(
            self._bg_max, self._bg_min, self._params.background.polarity,
        )

    def _render_current(self) -> None:
        if not self._frames or self._background is None:
            return
        if self._view_mode == "detection_bg":
            self._render_static_image(self._background, label="Detection bg")
            return
        if self._view_mode == "trail_bg" and self._trail is not None:
            self._render_static_image(self._trail, label="Trail bg")
            return
        frame = self._frames[self._current]
        dp = self._params.detection
        detections, fg_mask = detect_blobs_with_mask(
            frame, self._background, self._mask,
            threshold=dp.threshold, min_area=dp.min_area,
            max_area=dp.max_area, morph_kernel=dp.morph_kernel,
            dilate_iters=dp.dilate_iters, erode_iters=dp.erode_iters,
            min_solidity=dp.min_solidity, max_axis_ratio=dp.max_axis_ratio,
        )
        overlay = render_overlay(
            frame, self._sidecar.arena, detections,
            foreground_mask=fg_mask if self._show_mask else None,
            show_arena=self._show_arena,
            show_contours=self._show_contours,
            show_ellipses=self._show_ellipses,
            show_centroids=self._show_centroids,
        )
        self._set_pixmap(overlay)
        self._update_stats(detections)

    def _set_pixmap(self, bgr: np.ndarray) -> None:
        rgb = bgr[..., ::-1].copy()
        h, w, _ = rgb.shape
        image = QtGui.QImage(
            rgb.tobytes(), w, h, w * 3, QtGui.QImage.Format.Format_RGB888,
        )
        pix = QtGui.QPixmap.fromImage(image)
        target = self._frame_label.size()
        self._frame_label.setPixmap(
            pix.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _update_stats(
        self,
        detections: list[Detection],
        *,
        showing_label: str | None = None,
    ) -> None:
        if not self._frames:
            return
        idx = self._frame_indices[self._current] if self._frame_indices else -1
        self._frame_index_label.setText(f"{self._current + 1} / {len(self._frames)}")
        self._lbl_frame.setText(
            f"Frame: {self._current + 1} / {len(self._frames)}  (source idx {idx})",
        )
        if showing_label is not None:
            self._lbl_this.setText(f"Showing {showing_label}")
        else:
            self._lbl_this.setText(f"This frame: {len(detections)} detections")
        counts = self._counts_across_sample()
        if counts:
            self._lbl_agg.setText(self._summarise_counts(counts))
        self._lbl_shape.setText(self._summarise_shape(detections))

    def _render_static_image(self, gray: np.ndarray, *, label: str) -> None:
        overlay = render_overlay(
            gray, self._sidecar.arena, [],
            show_arena=self._show_arena,
            show_contours=False, show_ellipses=False, show_centroids=False,
        )
        self._set_pixmap(overlay)
        self._update_stats(detections=[], showing_label=label)

    def _counts_across_sample(self) -> list[int]:
        if self._background is None:
            return []
        dp = self._params.detection
        out: list[int] = []
        for frame in self._frames:
            dets, _ = detect_blobs_with_mask(
                frame, self._background, self._mask,
                threshold=dp.threshold, min_area=dp.min_area,
                max_area=dp.max_area, morph_kernel=dp.morph_kernel,
                dilate_iters=dp.dilate_iters, erode_iters=dp.erode_iters,
                min_solidity=dp.min_solidity, max_axis_ratio=dp.max_axis_ratio,
                keep_contour=False,
            )
            out.append(len(dets))
        return out

    def _summarise_shape(self, detections: list[Detection]) -> str:
        if not detections:
            return "Shape: no detections this frame"
        sols = [d.solidity for d in detections]
        ratios = [
            d.major_axis_px / d.minor_axis_px
            for d in detections if d.minor_axis_px > 0
        ]
        sol_med = statistics.median(sols)
        sol_min = min(sols)
        ratio_med = statistics.median(ratios) if ratios else 0.0
        ratio_max = max(ratios) if ratios else 0.0
        return (
            f"Shape: solidity median {sol_med:.2f} (min {sol_min:.2f}), "
            f"axis-ratio median {ratio_med:.1f} (max {ratio_max:.1f})"
        )

    def _summarise_counts(self, counts: list[int]) -> str:
        mean = statistics.fmean(counts)
        sd = statistics.pstdev(counts) if len(counts) > 1 else 0.0
        return (
            f"Across sample (n={len(counts)}): "
            f"{mean:.1f} ± {sd:.1f}  ({min(counts)} – {max(counts)})"
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._frames and self._background is not None:
            self._render_current()


def run(
    video: Path,
    sidecar_path: Path | None = None,
    params_path: Path | None = None,
) -> int:
    """Launch the GUI event loop. Returns a process exit code."""
    sidecar_path = sidecar_path or default_sidecar_path(video)
    if not sidecar_path.exists():
        print(
            f"Sidecar not found: {sidecar_path}\n"
            "Run pylace-annotate on this video first.",
            file=sys.stderr,
        )
        return 2
    sidecar = read_sidecar(sidecar_path)

    if sidecar.video.sha256 != video_sha256(video):
        print(
            "Warning: video SHA-256 differs from the sidecar; geometry may "
            "no longer be valid.", file=sys.stderr,
        )

    out_params = params_path or default_params_path(video)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = TuneWindow(video=video, sidecar=sidecar, params_path=out_params)
    window.resize(1100, 720)
    window.show()
    return app.exec()


__all__ = ["TuneWindow", "run"]

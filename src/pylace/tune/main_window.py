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

        from pylace.roi.geometry import ROISet
        self._arena_mask = arena_mask(sidecar.arena, sidecar.video.frame_size)
        self._roi_mask: np.ndarray | None = None
        self._roi_set: ROISet | None = None
        self._mask = self._arena_mask
        self._load_roi_mask_from_disk()
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
        self._show_roi = True
        self._show_contours = False
        self._show_ellipses = True
        self._show_centroids = True
        self._show_numbers = True
        self._view_mode: str = "frame"  # "frame" | "detection_bg" | "trail_bg"

        self._aggregate_cache_key: tuple | None = None
        self._aggregate_cache: dict[str, list[float]] | None = None
        self._dirty = False

        self.setWindowTitle(f"pylace-tune — {video.name}")
        self._build_ui()
        self._install_frame_shortcuts()
        self._initial_load()

    def _install_frame_shortcuts(self) -> None:
        """Window-level Left / Right shortcuts for frame stepping."""
        prev_action = QtGui.QAction("Previous frame", self)
        prev_action.setShortcut(QtGui.QKeySequence(Qt.Key.Key_Left))
        prev_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        prev_action.triggered.connect(lambda: self._step_frame(-1))
        self.addAction(prev_action)

        next_action = QtGui.QAction("Next frame", self)
        next_action.setShortcut(QtGui.QKeySequence(Qt.Key.Key_Right))
        next_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        next_action.triggered.connect(lambda: self._step_frame(1))
        self.addAction(next_action)

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

    def _load_roi_mask_from_disk(self) -> None:
        """Pick up <video>.pylace_rois.json if present so the preview honours it."""
        from pylace.roi.mask import build_combined_mask, build_split_masks
        from pylace.roi.sidecar import (
            ROISidecarSchemaError,
            default_rois_path,
            read_rois,
        )

        path = default_rois_path(self._video)
        if not path.exists():
            self._roi_mask = None
            self._roi_set = None
            self._mask = self._arena_mask
            return
        try:
            sidecar = read_rois(path)
        except (ROISidecarSchemaError, ValueError, OSError):
            self._roi_mask = None
            self._roi_set = None
            self._mask = self._arena_mask
            return
        self._roi_set = sidecar.roi_set
        fs = self._sidecar.video.frame_size
        if sidecar.roi_set.mode == "split":
            pairs = build_split_masks(sidecar.roi_set, fs)
            if pairs:
                # OR all split masks for the live preview; the actual
                # pylace-detect run still honours per-ROI splitting.
                combined = pairs[0][1].copy()
                for _, m in pairs[1:]:
                    combined |= m
            else:
                combined = None
        elif sidecar.roi_set.is_empty():
            combined = None
        else:
            combined = build_combined_mask(sidecar.roi_set, fs)
        self._roi_mask = combined
        if combined is not None:
            self._mask = self._arena_mask & combined
        else:
            self._mask = self._arena_mask

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

        self._btn_prev = QtWidgets.QToolButton(wrap)
        self._btn_prev.setText("◀")
        self._btn_prev.setToolTip("Previous frame (Left arrow)")
        self._btn_prev.clicked.connect(lambda: self._step_frame(-1))
        nav.addWidget(self._btn_prev)

        self._frame_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal, wrap)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(0)
        self._frame_slider.valueChanged.connect(self._on_frame_changed)
        nav.addWidget(self._frame_slider, stretch=1)

        self._btn_next = QtWidgets.QToolButton(wrap)
        self._btn_next.setText("▶")
        self._btn_next.setToolTip("Next frame (Right arrow)")
        self._btn_next.clicked.connect(lambda: self._step_frame(1))
        nav.addWidget(self._btn_next)

        self._frame_index_label = QtWidgets.QLabel("0 / 0", wrap)
        nav.addWidget(self._frame_index_label)
        v.addLayout(nav)

        return wrap

    def _step_frame(self, delta: int) -> None:
        if not self._frames:
            return
        target = max(0, min(len(self._frames) - 1, self._current + int(delta)))
        if target == self._current:
            return
        self._frame_slider.setValue(target)

    def _build_right_panel(self) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget(self)
        wrap.setMinimumWidth(360)
        v = QtWidgets.QVBoxLayout(wrap)
        v.setContentsMargins(4, 4, 4, 4)

        tabs = QtWidgets.QTabWidget(wrap)
        tabs.addTab(self._build_detection_page(tabs), "Detection")
        tabs.addTab(self._build_background_page(tabs), "Background")
        tabs.addTab(self._build_tracking_page(tabs), "Tracking")
        tabs.addTab(self._build_view_page(tabs), "View")
        v.addWidget(tabs)

        v.addWidget(self._h_separator(wrap))
        v.addWidget(self._build_stats_group(wrap))
        v.addWidget(self._h_separator(wrap))

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

        edit_rois = QtWidgets.QPushButton("Edit ROIs…", wrap)
        edit_rois.setToolTip(
            "Open the ROI editor as a dialog. The same widget as "
            "pylace-roi; on Apply the ROI sidecar is saved next to the "
            "video and the live preview re-renders against the new mask.",
        )
        edit_rois.clicked.connect(self._on_edit_rois)
        v.addWidget(edit_rois)

        return wrap

    def _h_separator(self, parent: QtWidgets.QWidget) -> QtWidgets.QFrame:
        line = QtWidgets.QFrame(parent)
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        return line

    def _build_detection_page(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(parent)
        form = QtWidgets.QFormLayout(page)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        dp = self._params.detection
        self._sb_threshold = self._spin(
            form, "Threshold", dp.threshold, 0, 255,
            tip="Pixel intensity below which a foreground pixel survives "
                "after background subtraction. Lower threshold = more "
                "permissive (more candidate blobs).",
        )
        self._sb_min_area = self._spin(
            form, "Min area (px)", dp.min_area, 1, 1_000_000,
            tip="Smallest contour kept. Below this, blobs are dust / noise / "
                "shadows of legs.",
        )
        self._sb_max_area = self._spin(
            form, "Max area (px)", dp.max_area, 1, 1_000_000,
            tip="Largest contour kept. Above this, the blob is two animals "
                "merged into a chain or part of the arena rim.",
        )
        self._sb_morph = self._spin(
            form, "Morph kernel", dp.morph_kernel, 0, 21,
            tip="Square kernel (px) used by the post-threshold open/close "
                "morphology pass. Smooths jagged boundaries.",
        )
        self._sb_dilate = self._spin(
            form, "Dilate iters", dp.dilate_iters, 0, 20,
            tip="How many dilate passes to run after threshold. Use to "
                "merge adjacent fragments of the same fly into one blob.",
        )
        self._sb_erode = self._spin(
            form, "Erode iters", dp.erode_iters, 0, 20,
            tip="How many erode passes to run after threshold. Use to "
                "break two flies that are nearly touching.",
        )
        self._sb_min_sol = self._dspin(
            form, "Min solidity (0=off)", dp.min_solidity, 0.0, 1.0, 0.01,
            tip="Solidity = area / convex-hull area. 0 disables the filter. "
                "~0.85 rejects diffuse shadows; lower (~0.7) for side-lying "
                "flies whose silhouette is less convex.",
        )
        self._sb_max_axis = self._dspin(
            form, "Max axis ratio (0=off)", dp.max_axis_ratio, 0.0, 30.0, 0.5,
            tip="major / minor axis ratio. 0 disables. ~5.0 rejects long thin "
                "streaks (wing reflections, antennae artefacts).",
        )
        for sb in (
            self._sb_threshold, self._sb_min_area, self._sb_max_area,
            self._sb_morph, self._sb_dilate, self._sb_erode,
        ):
            sb.valueChanged.connect(self._on_detection_changed)
        for dsb in (self._sb_min_sol, self._sb_max_axis):
            dsb.valueChanged.connect(self._on_detection_changed)
        return page

    def _build_background_page(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(parent)
        form = QtWidgets.QFormLayout(page)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        bp = self._params.background
        self._sb_bg_n = self._spin(
            form, "n_frames", bp.n_frames, 1, 1000,
            tip="Number of frames sampled to build the projection pair "
                "(per-pixel max + min). More frames = more robust bg, slower.",
        )
        self._sb_bg_start = self._dspin(
            form, "start_frac", bp.start_frac, 0.0, 1.0, 0.01,
            tip="Fraction of the video where bg sampling begins (0..1). "
                "Skip the first chunk if the camera is settling.",
        )
        self._sb_bg_end = self._dspin(
            form, "end_frac", bp.end_frac, 0.0, 1.0, 0.01,
            tip="Fraction of the video where bg sampling ends (0..1).",
        )

        self._cb_polarity = QtWidgets.QComboBox(page)
        self._cb_polarity.addItem("Dark on light (max → detection)", "dark_on_light")
        self._cb_polarity.addItem("Light on dark (min → detection)", "light_on_dark")
        idx = 0 if bp.polarity == "dark_on_light" else 1
        self._cb_polarity.setCurrentIndex(idx)
        self._cb_polarity.currentIndexChanged.connect(self._on_polarity_changed)
        self._add_form_row(
            form, "Polarity", self._cb_polarity,
            tip="Dark animals on a bright arena → max-projection feeds "
                "detection. Bright animals on a dark arena → min-projection. "
                "Both projections are always saved; the unused one becomes "
                "the trail/heatmap image.",
        )

        rebuild = QtWidgets.QPushButton("Rebuild background", page)
        rebuild.clicked.connect(self._on_rebuild_background)
        form.addRow(rebuild)
        return page

    def _build_tracking_page(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(parent)
        form = QtWidgets.QFormLayout(page)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        tp = self._params.tracking

        info = QtWidgets.QLabel(
            "These knobs are not applied in the live preview (tracking is "
            "sequential).\nThey are saved into the params sidecar and used by "
            "the next pylace-detect run.",
            page,
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: gray; font-style: italic;")
        form.addRow(info)

        self._cb_track_enabled = QtWidgets.QCheckBox("Enable Hungarian tracker")
        self._cb_track_enabled.setChecked(tp.enabled)
        self._add_form_row(
            form, "", self._cb_track_enabled,
            tip="Off = pylace-detect emits raw detections without track IDs. "
                "On = each detection gets a track_id assigned via Hungarian "
                "centroid matching across frames.",
        )

        self._sb_n_animals = self._spin(
            form, "N animals (0 = auto)", tp.n_animals or 0, 0, 100,
            tip="Set to the known animal count for fixed-N mode (the LACE-"
                "paper assumption). In fixed-N mode tracks never die during "
                "occlusion. 0 = dynamic-N: tracks are born when a detection "
                "appears and retired after Max missed frames.",
        )
        self._sb_track_dist = self._dspin(
            form, "Max distance (px)", tp.max_distance_px, 0.0, 5000.0, 5.0,
            tip="In dynamic-N mode, two same-track positions farther apart "
                "than this in successive frames cannot be matched. Ignored "
                "in fixed-N mode (the N animals must be matched to N tracks).",
        )
        self._sb_track_missed = self._spin(
            form, "Max missed frames", tp.max_missed_frames, 0, 1000,
            tip="In dynamic-N mode, a track unmatched for more than this "
                "many frames is retired. Ignored in fixed-N mode.",
        )
        self._sb_expected_area = self._dspin(
            form, "Expected animal area px (0 = auto)",
            tp.expected_animal_area_px or 0.0, 0.0, 1_000_000.0, 50.0,
            tip="Chain splitter cuts a blob whose area exceeds 1.5× this "
                "value perpendicular to its major axis (LACE Problem 5/6/7). "
                "0 = auto-learn the median from the first ~50 frames at "
                "runtime.",
        )
        self._sb_area_w = self._dspin(
            form, "Area cost weight", tp.area_cost_weight, 0.0, 10.0, 0.01,
            tip="Adds |Δarea|·w to the Hungarian cost so size differences "
                "tip ambiguous matches the right way. Try 0.05.",
        )
        self._sb_per_w = self._dspin(
            form, "Perimeter cost weight", tp.perimeter_cost_weight, 0.0, 10.0, 0.01,
            tip="Adds |Δperimeter|·w to the Hungarian cost. Useful when "
                "two animals have similar areas but very different shapes.",
        )

        self._cb_track_enabled.toggled.connect(self._on_tracking_changed)
        for sb in (self._sb_n_animals, self._sb_track_missed):
            sb.valueChanged.connect(self._on_tracking_changed)
        for dsb in (
            self._sb_track_dist, self._sb_expected_area,
            self._sb_area_w, self._sb_per_w,
        ):
            dsb.valueChanged.connect(self._on_tracking_changed)
        return page

    def _build_view_page(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(parent)
        v = QtWidgets.QVBoxLayout(page)

        prev_box = QtWidgets.QGroupBox("Preview window", page)
        prev_form = QtWidgets.QFormLayout(prev_box)
        prev_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._sb_prev_start = self._dspin(
            prev_form, "Start (s)", self._preview_start_s,
            0.0, self._video_duration_s, 1.0,
            tip="Earliest source-time the tuner samples preview frames from.",
        )
        self._sb_prev_end = self._dspin(
            prev_form, "End (s)", self._preview_end_s,
            0.0, self._video_duration_s, 1.0,
            tip="Latest source-time the tuner samples preview frames from.",
        )
        self._sb_prev_n = self._spin(
            prev_form, "Sample N frames", self._preview_n, 1, 200,
            tip="Number of evenly-spaced preview frames pulled from the "
                "Start..End window. Larger = better Stats average but slower.",
        )
        resample = QtWidgets.QPushButton("Resample", prev_box)
        resample.clicked.connect(self._on_resample)
        prev_form.addRow(resample)
        v.addWidget(prev_box)

        disp_box = QtWidgets.QGroupBox("Display", page)
        dv = QtWidgets.QVBoxLayout(disp_box)

        view_row = QtWidgets.QHBoxLayout()
        view_row.addWidget(QtWidgets.QLabel("Show:", disp_box))
        self._cb_view = QtWidgets.QComboBox(disp_box)
        self._cb_view.addItem("Sample frame", "frame")
        self._cb_view.addItem("Detection bg", "detection_bg")
        self._cb_view.addItem("Trail bg (animal heatmap)", "trail_bg")
        self._cb_view.currentIndexChanged.connect(self._on_view_mode_changed)
        self._cb_view.setToolTip(
            "Sample frame = the currently-selected frame from the preview "
            "window. Detection bg = the projection that detection subtracts. "
            "Trail bg = the opposite projection — where the animal lives "
            "most of the time.",
        )
        view_row.addWidget(self._cb_view, stretch=1)
        view_row.addWidget(self._help_button(self._cb_view.toolTip()))
        dv.addLayout(view_row)

        self._cb_mask = self._check(
            dv, "Foreground mask tint", self._show_mask,
            tip="Tints the pixels that survive threshold + morphology, so "
                "you can see what the contour finder will see.",
        )
        self._cb_arena = self._check(
            dv, "Arena outline", self._show_arena,
            tip="Draws the calibrated arena boundary on the overlay.",
        )
        self._cb_roi = self._check(
            dv, "ROI outline", self._show_roi,
            tip="Draws the ROI sidecar's outlines on the overlay (green for "
                "add ROIs, red for subtract, teal for the freehand mask). "
                "The detection footprint is always arena ∩ ROI; this toggle "
                "just controls visibility.",
        )
        self._cb_contours = self._check(
            dv, "Contours", self._show_contours,
            tip="Draws each detection's raw OpenCV contour.",
        )
        self._cb_ellipses = self._check(
            dv, "Ellipses", self._show_ellipses,
            tip="Draws the fitted ellipse for each detection (LACE-style "
                "ellipse hypothesis).",
        )
        self._cb_centroids = self._check(
            dv, "Centroids", self._show_centroids,
            tip="Marks each detection's centroid with a coloured pixel.",
        )
        self._cb_numbers = self._check(
            dv, "Detection numbers", self._show_numbers,
            tip="Labels each detection with its track ID (when tracking has "
                "run) or its per-frame index. Numbers match the entries in "
                "the Detection info panel below.",
        )
        for cb in (
            self._cb_mask, self._cb_arena, self._cb_roi, self._cb_contours,
            self._cb_ellipses, self._cb_centroids, self._cb_numbers,
        ):
            cb.toggled.connect(self._on_display_toggled)
        v.addWidget(disp_box)
        v.addStretch(1)
        return page

    def _build_stats_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Detection info", parent)
        v = QtWidgets.QVBoxLayout(box)
        self._lbl_info = QtWidgets.QLabel("—", box)
        mono = QtGui.QFont("Monospace")
        mono.setStyleHint(QtGui.QFont.StyleHint.TypeWriter)
        mono.setPointSize(9)
        self._lbl_info.setFont(mono)
        self._lbl_info.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse,
        )
        self._lbl_info.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
        )
        self._lbl_info.setMinimumHeight(180)
        v.addWidget(self._lbl_info)
        return box

    # ── Widget builders ────────────────────────────────────────────────

    def _spin(
        self, form: QtWidgets.QFormLayout, label: str, value: int, lo: int, hi: int,
        *, tip: str | None = None,
    ) -> QtWidgets.QSpinBox:
        sb = QtWidgets.QSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(int(value))
        self._add_form_row(form, label, sb, tip=tip)
        return sb

    def _dspin(
        self, form: QtWidgets.QFormLayout, label: str,
        value: float, lo: float, hi: float, step: float,
        *, tip: str | None = None,
    ) -> QtWidgets.QDoubleSpinBox:
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setSingleStep(step)
        sb.setDecimals(3)
        sb.setValue(float(value))
        self._add_form_row(form, label, sb, tip=tip)
        return sb

    def _check(
        self, layout: QtWidgets.QVBoxLayout, label: str, value: bool,
        *, tip: str | None = None,
    ) -> QtWidgets.QCheckBox:
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        cb = QtWidgets.QCheckBox(label)
        cb.setChecked(value)
        h.addWidget(cb)
        if tip:
            cb.setToolTip(tip)
            h.addWidget(self._help_button(tip))
        h.addStretch(1)
        layout.addWidget(row)
        return cb

    def _add_form_row(
        self,
        form: QtWidgets.QFormLayout,
        label: str,
        field: QtWidgets.QWidget,
        *,
        tip: str | None = None,
    ) -> None:
        """Add ``[label] [?] field`` to a QFormLayout. ``?`` only appears if tip."""
        if not tip:
            if label:
                form.addRow(label, field)
            else:
                form.addRow(field)
            return
        field.setToolTip(tip)
        label_widget = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(label_widget)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(2)
        if label:
            h.addWidget(QtWidgets.QLabel(label))
        h.addWidget(self._help_button(tip))
        h.addStretch(1)
        form.addRow(label_widget, field)

    def _help_button(self, tip: str) -> QtWidgets.QToolButton:
        """A small clickable ``?`` that pops the given tip near the cursor."""
        btn = QtWidgets.QToolButton()
        btn.setText("?")
        btn.setToolTip(tip)
        btn.setAutoRaise(True)
        btn.setCursor(Qt.CursorShape.WhatsThisCursor)
        btn.setStyleSheet(
            "QToolButton {"
            "  color: #2a78c2; font-weight: bold; font-size: 10pt;"
            "  border: 1px solid #2a78c2; border-radius: 8px;"
            "  min-width: 16px; max-width: 16px;"
            "  min-height: 16px; max-height: 16px;"
            "  padding: 0px;"
            "}"
            "QToolButton:hover { background-color: #e6f0fa; }"
        )
        btn.clicked.connect(
            lambda _checked=False, t=tip: QtWidgets.QToolTip.showText(
                QtGui.QCursor.pos(), t,
            )
        )
        return btn

    # ── Event handlers ─────────────────────────────────────────────────

    def _mark_dirty(self) -> None:
        if not self._dirty:
            self._dirty = True
            base = self.windowTitle()
            if not base.startswith("● "):
                self.setWindowTitle(f"● {base}")

    def _mark_clean(self) -> None:
        self._dirty = False
        base = self.windowTitle()
        if base.startswith("● "):
            self.setWindowTitle(base[2:])

    def _on_detection_changed(self) -> None:
        self._mark_dirty()
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
        self._invalidate_aggregate_cache()
        self._render_current()

    def _on_tracking_changed(self) -> None:
        self._mark_dirty()
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
        self._show_roi = self._cb_roi.isChecked()
        self._show_contours = self._cb_contours.isChecked()
        self._show_ellipses = self._cb_ellipses.isChecked()
        self._show_centroids = self._cb_centroids.isChecked()
        self._show_numbers = self._cb_numbers.isChecked()
        self._render_current()

    def _on_view_mode_changed(self, _index: int) -> None:
        self._view_mode = self._cb_view.currentData() or "frame"
        self._render_current()

    def _on_polarity_changed(self, _index: int) -> None:
        self._mark_dirty()
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
        self._mark_dirty()
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

    def _on_save_params(self) -> bool:
        try:
            write_params(
                self._params,
                video_path=self._video,
                video_sha256_hex=self._sidecar.video.sha256,
                out_path=self._params_path,
            )
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Save failed", f"Could not write params:\n{exc}",
            )
            return False
        self._mark_clean()
        self.statusBar().showMessage(
            f"Saved params to {self._params_path.name}", 5000,
        )
        return True

    def _on_edit_rois(self) -> None:
        """Pop the ROI editor as a modal dialog; reload + re-render on apply."""
        from pylace.roi.panel import RoiEditDialog
        from pylace.roi.sidecar import default_rois_path

        dlg = RoiEditDialog(
            video=self._video,
            rois_path=default_rois_path(self._video),
            arena_sidecar=self._sidecar,
            parent=self,
        )
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        # The dialog has saved on Apply. Reload + re-render against the
        # new mask, and invalidate the aggregate cache since the
        # detection footprint changed.
        self._load_roi_mask_from_disk()
        self._invalidate_aggregate_cache()
        self._render_current()
        self.statusBar().showMessage("ROI mask updated.", 4000)

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
        # Saving a preset to a *different* path doesn't update the
        # canonical sidecar, so we don't clear the dirty flag — the
        # main params file may still be out of date.
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
        # The loaded preset is, by definition, what's now on disk in
        # ``params_path`` only if path_str == self._params_path. We can't
        # know cheaply, so leave the dirty flag in its current state and
        # let the user save explicitly if they want to apply.
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

    def _invalidate_aggregate_cache(self) -> None:
        self._aggregate_cache_key = None
        self._aggregate_cache = None

    def _sample_frames(self) -> None:
        start_frame = int(round(self._preview_start_s * self._fps))
        end_frame = int(round(self._preview_end_s * self._fps))
        end_frame = min(end_frame, self._video_total_frames)
        end_frame = max(end_frame, start_frame + 1)
        self._frames, self._frame_indices = sample_preview_frames(
            self._video, n=self._preview_n,
            start_frame=start_frame, end_frame=end_frame,
        )
        self._invalidate_aggregate_cache()
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
        self._invalidate_aggregate_cache()

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
            roi_set=self._roi_set,
            show_arena=self._show_arena,
            show_roi=self._show_roi,
            show_contours=self._show_contours,
            show_ellipses=self._show_ellipses,
            show_centroids=self._show_centroids,
            show_numbers=self._show_numbers,
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
        self._lbl_info.setText(
            self._format_info_block(
                detections, source_idx=idx, showing_label=showing_label,
            )
        )

    def _render_static_image(self, gray: np.ndarray, *, label: str) -> None:
        overlay = render_overlay(
            gray, self._sidecar.arena, [],
            roi_set=self._roi_set,
            show_arena=self._show_arena,
            show_roi=self._show_roi,
            show_contours=False, show_ellipses=False, show_centroids=False,
        )
        self._set_pixmap(overlay)
        self._update_stats(detections=[], showing_label=label)

    def _aggregate_across_sample(self) -> dict[str, list[float]] | None:
        """Run detection on every sample frame; return per-detection feature lists."""
        if self._background is None or not self._frames:
            return None
        dp = self._params.detection
        key = (
            id(self._frames), id(self._background),
            dp.threshold, dp.min_area, dp.max_area, dp.morph_kernel,
            dp.dilate_iters, dp.erode_iters,
            dp.min_solidity, dp.max_axis_ratio,
        )
        if self._aggregate_cache_key == key and self._aggregate_cache is not None:
            return self._aggregate_cache

        counts: list[int] = []
        areas: list[float] = []
        majors: list[float] = []
        minors: list[float] = []
        sols: list[float] = []
        ratios: list[float] = []
        for frame in self._frames:
            dets, _ = detect_blobs_with_mask(
                frame, self._background, self._mask,
                threshold=dp.threshold, min_area=dp.min_area,
                max_area=dp.max_area, morph_kernel=dp.morph_kernel,
                dilate_iters=dp.dilate_iters, erode_iters=dp.erode_iters,
                min_solidity=dp.min_solidity, max_axis_ratio=dp.max_axis_ratio,
                keep_contour=False,
            )
            counts.append(len(dets))
            for d in dets:
                areas.append(d.area_px)
                majors.append(d.major_axis_px)
                minors.append(d.minor_axis_px)
                sols.append(d.solidity)
                if d.minor_axis_px > 0:
                    ratios.append(d.major_axis_px / d.minor_axis_px)
        out = {
            "counts": counts, "areas": areas, "majors": majors,
            "minors": minors, "sols": sols, "ratios": ratios,
        }
        self._aggregate_cache_key = key
        self._aggregate_cache = out
        return out

    def _format_info_block(
        self,
        detections: list[Detection],
        *,
        source_idx: int,
        showing_label: str | None,
    ) -> str:
        lines: list[str] = []
        lines.append(
            f"Frame {self._current + 1} / {len(self._frames)}  "
            f"(source idx {source_idx})",
        )
        if showing_label is not None:
            lines.append(f"Showing: {showing_label}")
            return "\n".join(lines)

        lines.append(f"This frame: {len(detections)} detections")
        if detections:
            row = "  {:>3}  {:>6}  {:>6}  {:>6}  {:>5}  {:>5}"
            lines.append(row.format("idx", "area", "major", "minor", "solid", "ratio"))
            for i, d in enumerate(detections):
                label = d.track_id if d.track_id >= 0 else i
                ratio = (
                    d.major_axis_px / d.minor_axis_px
                    if d.minor_axis_px > 0 else 0.0
                )
                lines.append(row.format(
                    str(label),
                    f"{d.area_px:.0f}",
                    f"{d.major_axis_px:.1f}",
                    f"{d.minor_axis_px:.1f}",
                    f"{d.solidity:.2f}",
                    f"{ratio:.1f}",
                ))

        agg = self._aggregate_across_sample()
        if agg and agg["counts"]:
            lines.append("")
            lines.append(
                f"Across sample (n={len(agg['counts'])} frames, "
                f"{len(agg['areas'])} detections):",
            )
            lines.append(self._fmt_count(agg["counts"]))
            if agg["areas"]:
                lines.append(self._fmt_mean("area  ", agg["areas"], unit="px²"))
                lines.append(self._fmt_mean("major ", agg["majors"], unit="px"))
                lines.append(self._fmt_mean("minor ", agg["minors"], unit="px"))
                lines.append(self._fmt_mean("solid ", agg["sols"], unit=""))
            if agg["ratios"]:
                lines.append(self._fmt_mean("ratio ", agg["ratios"], unit=""))
        return "\n".join(lines)

    def _fmt_count(self, counts: list[int]) -> str:
        mean = statistics.fmean(counts)
        sd = statistics.pstdev(counts) if len(counts) > 1 else 0.0
        return (
            f"  count  {mean:6.1f} ± {sd:5.1f}   "
            f"({min(counts)} – {max(counts)})"
        )

    def _fmt_mean(self, label: str, xs: list[float], *, unit: str) -> str:
        if not xs:
            return f"  {label}     —"
        mean = statistics.fmean(xs)
        sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        u = f" {unit}" if unit else ""
        return (
            f"  {label} {mean:6.1f} ± {sd:5.1f}{u}   "
            f"({min(xs):.1f} – {max(xs):.1f})"
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._frames and self._background is not None:
            self._render_current()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        if not self._dirty:
            super().closeEvent(event)
            return
        choice = QtWidgets.QMessageBox.question(
            self,
            "Save tuning changes?",
            "You have unsaved tuning changes. Save them to "
            f"{self._params_path.name} before quitting?",
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Save,
        )
        if choice == QtWidgets.QMessageBox.StandardButton.Save:
            if self._on_save_params():
                super().closeEvent(event)
            else:
                event.ignore()
        elif choice == QtWidgets.QMessageBox.StandardButton.Discard:
            super().closeEvent(event)
        else:
            event.ignore()


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

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
from pylace.detect.background import build_max_projection_background
from pylace.detect.frame import Detection, detect_blobs_with_mask
from pylace.tune.frame_loader import sample_preview_frames
from pylace.tune.overlay import render_overlay
from pylace.tune.params import (
    BackgroundParams,
    DetectionParams,
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
        self._rebuild_background()
        self._render_current()

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
        v.addWidget(self._build_preview_group(wrap))
        v.addWidget(self._build_display_group(wrap))
        v.addWidget(self._build_stats_group(wrap))
        v.addStretch(1)

        save = QtWidgets.QPushButton("Save params", wrap)
        save.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        save.clicked.connect(self._on_save_params)
        v.addWidget(save)

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
        for sb in (self._sb_threshold, self._sb_min_area, self._sb_max_area, self._sb_morph):
            sb.valueChanged.connect(self._on_detection_changed)
        return box

    def _build_background_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Background (max-projection)", parent)
        form = QtWidgets.QFormLayout(box)
        bp = self._params.background
        self._sb_bg_n = self._spin(form, "n_frames", bp.n_frames, 1, 1000)
        self._sb_bg_start = self._dspin(form, "start_frac", bp.start_frac, 0.0, 1.0, 0.01)
        self._sb_bg_end = self._dspin(form, "end_frac", bp.end_frac, 0.0, 1.0, 0.01)
        rebuild = QtWidgets.QPushButton("Rebuild background", box)
        rebuild.clicked.connect(self._on_rebuild_background)
        form.addRow(rebuild)
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
        v.addWidget(self._lbl_frame)
        v.addWidget(self._lbl_this)
        v.addWidget(self._lbl_agg)
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
            ),
            background=self._params.background,
        )
        self._render_current()

    def _on_display_toggled(self) -> None:
        self._show_mask = self._cb_mask.isChecked()
        self._show_arena = self._cb_arena.isChecked()
        self._show_contours = self._cb_contours.isChecked()
        self._show_ellipses = self._cb_ellipses.isChecked()
        self._show_centroids = self._cb_centroids.isChecked()
        self._render_current()

    def _on_rebuild_background(self) -> None:
        self._params = TuningParams(
            detection=self._params.detection,
            background=BackgroundParams(
                n_frames=self._sb_bg_n.value(),
                start_frac=float(self._sb_bg_start.value()),
                end_frac=float(self._sb_bg_end.value()),
            ),
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
        self._background = build_max_projection_background(
            self._video, n_frames=bp.n_frames,
            start_frac=bp.start_frac, end_frac=bp.end_frac,
        )

    def _render_current(self) -> None:
        if not self._frames or self._background is None:
            return
        frame = self._frames[self._current]
        dp = self._params.detection
        detections, fg_mask = detect_blobs_with_mask(
            frame, self._background, self._mask,
            threshold=dp.threshold, min_area=dp.min_area,
            max_area=dp.max_area, morph_kernel=dp.morph_kernel,
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

    def _update_stats(self, detections: list[Detection]) -> None:
        if not self._frames:
            return
        idx = self._frame_indices[self._current] if self._frame_indices else -1
        self._frame_index_label.setText(f"{self._current + 1} / {len(self._frames)}")
        self._lbl_frame.setText(
            f"Frame: {self._current + 1} / {len(self._frames)}  (source idx {idx})",
        )
        self._lbl_this.setText(f"This frame: {len(detections)} detections")
        counts = self._counts_across_sample()
        if counts:
            self._lbl_agg.setText(self._summarise_counts(counts))

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
                keep_contour=False,
            )
            out.append(len(dets))
        return out

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

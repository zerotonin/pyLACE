# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — roi.panel                                              ║
# ║  « reusable ROI-edit widget + modal dialog wrapper »             ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  RoiEditPanel is a self-contained QWidget hosting the shape /    ║
# ║  brush toolbar, the canvas, and the right-side ROI list. Both    ║
# ║  the standalone ``pylace-roi`` window and the ``Edit ROIs…``     ║
# ║  dialog launched from ``pylace-tune`` embed the same panel, so   ║
# ║  the feature set is identical regardless of how the user got     ║
# ║  there.                                                          ║
# ║                                                                  ║
# ║  RoiEditDialog wraps the panel in a QDialog with Apply (save +   ║
# ║  close) and Cancel buttons. On Apply the panel saves to its      ║
# ║  configured ``rois_path`` so the on-disk sidecar is canonical    ║
# ║  the moment the dialog returns Accepted; the caller can then     ║
# ║  reload and re-render its previews.                              ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Reusable ROI editor as a QWidget panel + a QDialog wrapper."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from pylace.annotator.sidecar import Sidecar
from pylace.detect.arena_mask import arena_mask
from pylace.detect.background import (
    default_background_paths,
    load_background_png,
)
from pylace.roi.auto_roi import AutoRoiParams, auto_rois_from_diff
from pylace.roi.canvas import RoiCanvas
from pylace.roi.geometry import ROI, ROIMode, ROISet
from pylace.roi.sidecar import (
    ROISidecar,
    ROISidecarSchemaError,
    read_rois,
    write_rois,
)


class RoiEditPanel(QtWidgets.QWidget):
    """Self-contained ROI editor: toolbar + canvas + ROI list.

    Embed inside a QMainWindow (standalone ``pylace-roi``) or a
    QDialog (``Edit ROIs…`` dialog from ``pylace-tune``). The panel
    owns the canvas, the in-memory ROISet, and the I/O helpers; the
    enclosing shell only needs to wire window-level shortcuts and
    chrome.
    """

    statusMessage = QtCore.pyqtSignal(str, int)
    """``(text, timeout_ms)`` — emitted when the panel wants to surface a
    transient message in the host's status bar."""

    dirtyChanged = QtCore.pyqtSignal(bool)
    """True when the in-memory ROISet has unsaved edits."""

    def __init__(
        self,
        video: Path,
        rois_path: Path,
        arena_sidecar: Sidecar | None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._video = video
        self._rois_path = rois_path
        self._arena_sidecar = arena_sidecar
        self._roi_set: ROISet = ROISet()
        self._dirty = False

        self._canvas = RoiCanvas(self)
        self._canvas.roiAdded.connect(self._on_roi_added)
        self._canvas.set_arena(arena_sidecar.arena if arena_sidecar else None)
        self._first_frame_rgb = self._load_first_frame()
        self._bg_max, self._bg_min = self._load_background_pair()
        self._view_mode: str = "frame"

        self._build_layout()
        self._load_existing_sidecar()
        self._apply_view_mode()

    # ── Public API ─────────────────────────────────────────────────────

    def roi_set(self) -> ROISet:
        return self._roi_set

    def set_roi_set(self, roi_set: ROISet) -> None:
        self._roi_set = roi_set
        self._refresh_list()
        self._canvas.set_roi_set(self._roi_set)
        self._sync_mode_radios()
        self._mark_clean()

    def is_dirty(self) -> bool:
        return self._dirty

    def rois_path(self) -> Path:
        return self._rois_path

    def save_to(self, path: Path) -> None:
        sha = self._arena_sidecar.video.sha256 if self._arena_sidecar else ""
        sidecar = ROISidecar(
            video_path=str(self._video),
            video_sha256=sha,
            roi_set=self._roi_set,
        )
        write_rois(sidecar, path)
        if path == self._rois_path:
            self._mark_clean()

    def load_from(self, path: Path) -> None:
        sidecar = read_rois(path)
        self._roi_set = sidecar.roi_set
        self._refresh_list()
        self._canvas.set_roi_set(self._roi_set)
        self._sync_mode_radios()
        self._mark_clean()

    # ── Layout ─────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self._build_toolbar())

        body = QtWidgets.QWidget(self)
        h = QtWidgets.QHBoxLayout(body)
        h.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea(body)
        scroll.setWidget(self._canvas)
        scroll.setWidgetResizable(False)
        h.addWidget(scroll, stretch=1)

        h.addWidget(self._build_right_panel(body))
        v.addWidget(body, stretch=1)

    def _build_toolbar(self) -> QtWidgets.QToolBar:
        bar = QtWidgets.QToolBar(self)
        bar.setMovable(False)
        # Use widget-context shortcuts so the keys still work inside a
        # QDialog (and don't fight the dialog's Esc-cancels-default).
        ctx = Qt.ShortcutContext.WidgetWithChildrenShortcut

        def _action(label: str, key: str | None, slot, tooltip: str = ""):
            act = QtGui.QAction(label, self)
            if key:
                act.setShortcut(QtGui.QKeySequence(key))
                act.setShortcutContext(ctx)
            if tooltip:
                act.setToolTip(tooltip)
            act.triggered.connect(slot)
            bar.addAction(act)
            self.addAction(act)
            return act

        _action("Circle", "C", lambda: self._canvas.set_tool("circle"),
                "Click-drag to draw a circular ROI (C).")
        _action("Rectangle", "R", lambda: self._canvas.set_tool("rectangle"),
                "Click-drag to draw a rectangular ROI (R).")
        _action("Polygon", "P", lambda: self._canvas.set_tool("polygon"),
                "Left-click vertices, right-click to close (P).")
        bar.addSeparator()
        _action("Brush", "B", lambda: self._canvas.set_tool("brush"),
                "Paint freehand pixels into the ROI mask (B).")
        _action("Eraser", "E", lambda: self._canvas.set_tool("eraser"),
                "Erase freehand pixels from the ROI mask (E).")

        bar.addWidget(QtWidgets.QLabel(" Brush size:"))
        self._sb_brush = QtWidgets.QSpinBox()
        self._sb_brush.setRange(1, 200)
        self._sb_brush.setValue(12)
        self._sb_brush.valueChanged.connect(self._canvas.set_brush_radius)
        bar.addWidget(self._sb_brush)

        bar.addSeparator()
        _action("No tool", None, lambda: self._canvas.set_tool(None),
                "Disable any active drawing / brush tool.")
        return bar

    def _build_right_panel(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(parent)
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)
        panel.setMinimumWidth(300)

        v.addWidget(self._build_view_group(panel))
        v.addWidget(self._build_auto_group(panel))
        v.addWidget(self._build_mode_group(panel))

        self._list = QtWidgets.QListWidget(panel)
        self._list.currentRowChanged.connect(self._on_list_selection_changed)
        v.addWidget(QtWidgets.QLabel("ROIs:", panel))
        v.addWidget(self._list, stretch=1)

        op_row = QtWidgets.QHBoxLayout()
        toggle = QtWidgets.QPushButton("Toggle op (add ↔ subtract)", panel)
        toggle.clicked.connect(self._on_toggle_op)
        op_row.addWidget(toggle)
        delete = QtWidgets.QPushButton("Delete", panel)
        delete.setShortcut(QtGui.QKeySequence("Del"))
        delete.clicked.connect(self._on_delete_selected)
        op_row.addWidget(delete)
        v.addLayout(op_row)

        save_row = QtWidgets.QHBoxLayout()
        save = QtWidgets.QPushButton("Save", panel)
        save.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        save.clicked.connect(self._on_save)
        save_row.addWidget(save)
        save_as = QtWidgets.QPushButton("Save as…", panel)
        save_as.clicked.connect(self._on_save_as)
        save_row.addWidget(save_as)
        load = QtWidgets.QPushButton("Load…", panel)
        load.clicked.connect(self._on_load)
        save_row.addWidget(load)
        v.addLayout(save_row)
        return panel

    def _build_view_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("View", parent)
        v = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Show:", box))
        self._cb_view = QtWidgets.QComboBox(box)
        self._cb_view.addItem("First frame", "frame")
        self._cb_view.addItem("Background max", "bg_max")
        self._cb_view.addItem("Background min", "bg_min")
        bg_available = self._bg_max is not None and self._bg_min is not None
        if not bg_available:
            for i in (1, 2):
                self._cb_view.model().item(i).setEnabled(False)
            self._cb_view.setToolTip(
                "Background pair not on disk. Run pylace-tune or pylace-detect "
                "on this video first to compute and save it.",
            )
        self._cb_view.currentIndexChanged.connect(self._on_view_changed)
        row.addWidget(self._cb_view, stretch=1)
        v.addLayout(row)
        return box

    def _build_auto_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Auto-ROI", parent)
        v = QtWidgets.QVBoxLayout(box)
        hint = QtWidgets.QLabel(
            "Generate polygons from the trail (max-vs-min bg diff). "
            "Erodes specks, then dilates so the ROI is slightly larger "
            "than the actual trail.",
            box,
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        v.addWidget(hint)
        self._btn_auto = QtWidgets.QPushButton("Generate auto-ROIs", box)
        self._btn_auto.clicked.connect(self._on_auto_roi)
        if self._bg_max is None or self._bg_min is None:
            self._btn_auto.setEnabled(False)
            self._btn_auto.setToolTip(
                "Background pair not on disk. Run pylace-tune or "
                "pylace-detect first.",
            )
        v.addWidget(self._btn_auto)
        return box

    def _build_mode_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Mode", parent)
        v = QtWidgets.QVBoxLayout(box)
        self._rb_merge = QtWidgets.QRadioButton(
            "Merge (single combined mask)", box,
        )
        self._rb_merge.setChecked(True)
        self._rb_merge.toggled.connect(
            lambda on: on and self._set_mode("merge"),
        )
        v.addWidget(self._rb_merge)
        self._rb_split = QtWidgets.QRadioButton(
            "Split (each ROI as its own sub-video)", box,
        )
        self._rb_split.setToolTip(
            "Each add-ROI runs detection independently and gets its own "
            "roi_label in the output CSV. Subtract ROIs are ignored in "
            "split mode (not meaningful as sub-videos).",
        )
        self._rb_split.toggled.connect(
            lambda on: on and self._set_mode("split"),
        )
        v.addWidget(self._rb_split)
        return box

    # ── Slots ─────────────────────────────────────────────────────────

    def _on_roi_added(self, roi: ROI) -> None:
        self._roi_set.add(roi)
        self._refresh_list()
        self._canvas.set_roi_set(self._roi_set)
        self._mark_dirty()
        self.statusMessage.emit(
            f"Added {_describe_roi(roi)} (#{len(self._roi_set.rois) - 1}).", 4000,
        )

    def _on_view_changed(self, _index: int) -> None:
        mode = self._cb_view.currentData()
        if mode in ("bg_max", "bg_min") and (
            self._bg_max is None or self._bg_min is None
        ):
            self.statusMessage.emit("No background pair on disk.", 4000)
            return
        self._view_mode = mode or "frame"
        self._apply_view_mode()

    def _on_auto_roi(self) -> None:
        if self._bg_max is None or self._bg_min is None:
            self.statusMessage.emit(
                "No background pair — run pylace-tune first.", 5000,
            )
            return
        if self._arena_sidecar is None:
            self.statusMessage.emit(
                "Auto-ROI needs an arena sidecar. Run pylace-annotate first.",
                5000,
            )
            return
        h, w = self._bg_max.shape
        a_mask = arena_mask(self._arena_sidecar.arena, frame_size=(w, h))
        rois = auto_rois_from_diff(
            self._bg_max, self._bg_min, a_mask, params=AutoRoiParams(),
        )
        if not rois:
            self.statusMessage.emit(
                "Auto-ROI found nothing — try recomputing the background "
                "or check polarity.", 6000,
            )
            return
        for roi in rois:
            self._roi_set.add(roi)
        self._refresh_list()
        self._canvas.set_roi_set(self._roi_set)
        self._mark_dirty()
        self.statusMessage.emit(
            f"Auto-ROI added {len(rois)} polygon(s).", 5000,
        )

    def _on_list_selection_changed(self, row: int) -> None:
        self._canvas.set_selected(row)

    def _on_toggle_op(self) -> None:
        index = self._list.currentRow()
        if index < 0 or index >= len(self._roi_set.rois):
            return
        roi = self._roi_set.rois[index]
        roi.operation = "subtract" if roi.operation == "add" else "add"
        self._refresh_list()
        self._list.setCurrentRow(index)
        self._canvas.set_roi_set(self._roi_set)
        self._mark_dirty()

    def _on_delete_selected(self) -> None:
        index = self._list.currentRow()
        if index < 0:
            return
        self._roi_set.remove_at(index)
        self._refresh_list()
        self._canvas.set_roi_set(self._roi_set)
        self._mark_dirty()

    def _on_save(self) -> None:
        try:
            self.save_to(self._rois_path)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Save failed", f"Could not write ROI sidecar:\n{exc}",
            )
            return
        self.statusMessage.emit(f"Saved to {self._rois_path.name}", 5000)

    def _on_save_as(self) -> None:
        path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save ROI preset",
            str(self._rois_path.parent / "preset.pylace_rois.json"),
            "pyLACE ROIs (*.json);;All files (*)",
        )
        if not path_str:
            return
        try:
            self.save_to(Path(path_str))
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Save failed", f"Could not write ROI sidecar:\n{exc}",
            )
            return
        self.statusMessage.emit(f"Saved to {Path(path_str).name}", 5000)

    def _on_load(self) -> None:
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load ROI preset", str(self._rois_path.parent),
            "pyLACE ROIs (*.json);;All files (*)",
        )
        if not path_str:
            return
        try:
            self.load_from(Path(path_str))
        except (ROISidecarSchemaError, ValueError, OSError) as exc:
            self.statusMessage.emit(f"Cannot load: {exc}", 7000)
            return
        self.statusMessage.emit(f"Loaded {Path(path_str).name}", 5000)

    def _set_mode(self, mode: ROIMode) -> None:
        if self._roi_set.mode != mode:
            self._roi_set.mode = mode
            self._mark_dirty()

    # ── Helpers ───────────────────────────────────────────────────────

    def _load_first_frame(self) -> np.ndarray:
        cap = cv2.VideoCapture(str(self._video))
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {self._video}")
        try:
            ok, bgr = cap.read()
            if not ok:
                raise OSError(f"Cannot read first frame of {self._video}.")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
        self._canvas.set_frame(rgb)
        return rgb

    def _load_background_pair(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        max_path, min_path = default_background_paths(self._video)
        if not (max_path.exists() and min_path.exists()):
            return None, None
        try:
            return load_background_png(max_path), load_background_png(min_path)
        except OSError:
            return None, None

    def _apply_view_mode(self) -> None:
        if self._view_mode == "bg_max" and self._bg_max is not None:
            self._canvas.set_frame(cv2.cvtColor(self._bg_max, cv2.COLOR_GRAY2RGB))
        elif self._view_mode == "bg_min" and self._bg_min is not None:
            self._canvas.set_frame(cv2.cvtColor(self._bg_min, cv2.COLOR_GRAY2RGB))
        else:
            self._canvas.set_frame(self._first_frame_rgb)

    def _load_existing_sidecar(self) -> None:
        if not self._rois_path.exists():
            return
        try:
            sidecar = read_rois(self._rois_path)
        except ROISidecarSchemaError as exc:
            self.statusMessage.emit(f"Existing ROI sidecar ignored: {exc}", 7000)
            return
        self._roi_set = sidecar.roi_set
        self._refresh_list()
        self._canvas.set_roi_set(self._roi_set)
        self._sync_mode_radios()
        # The on-disk file matches in-memory; not dirty.
        self._mark_clean()

    def _refresh_list(self) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for index, roi in enumerate(self._roi_set.rois):
            self._list.addItem(_format_list_row(index, roi))
        self._list.blockSignals(False)
        if self._roi_set.rois:
            self._list.setCurrentRow(
                min(self._list.count() - 1, max(0, self._list.currentRow())),
            )

    def _sync_mode_radios(self) -> None:
        if self._roi_set.mode == "split":
            self._rb_split.setChecked(True)
        else:
            self._rb_merge.setChecked(True)

    def _mark_dirty(self) -> None:
        if not self._dirty:
            self._dirty = True
            self.dirtyChanged.emit(True)

    def _mark_clean(self) -> None:
        if self._dirty:
            self._dirty = False
            self.dirtyChanged.emit(False)


# ─────────────────────────────────────────────────────────────────
#  Modal dialog wrapper
# ─────────────────────────────────────────────────────────────────


class RoiEditDialog(QtWidgets.QDialog):
    """Modal ROI editor — same panel as ``pylace-roi``, dialogue chrome.

    On Apply, saves to the panel's ``rois_path`` and returns
    ``Accepted``. On Cancel, returns ``Rejected``; in-memory edits
    are discarded but anything explicitly saved during the session
    via the panel's Save / Save as buttons stays on disk.
    """

    def __init__(
        self,
        video: Path,
        rois_path: Path,
        arena_sidecar: Sidecar | None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Edit ROIs — {video.name}")
        self._panel = RoiEditPanel(
            video=video, rois_path=rois_path,
            arena_sidecar=arena_sidecar, parent=self,
        )

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.addWidget(self._panel, stretch=1)

        self._status = QtWidgets.QLabel(self)
        self._status.setStyleSheet("color: #555;")
        v.addWidget(self._status)
        self._panel.statusMessage.connect(self._on_panel_status)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        apply_btn = QtWidgets.QPushButton("Apply (save and close)", self)
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self._on_apply)
        button_row.addWidget(apply_btn)
        cancel_btn = QtWidgets.QPushButton("Cancel", self)
        cancel_btn.clicked.connect(self.reject)
        button_row.addWidget(cancel_btn)
        v.addLayout(button_row)

        self.resize(1200, 760)

    def panel(self) -> RoiEditPanel:
        return self._panel

    def _on_apply(self) -> None:
        try:
            self._panel.save_to(self._panel.rois_path())
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Save failed",
                f"Could not write ROI sidecar:\n{exc}",
            )
            return
        self.accept()

    def _on_panel_status(self, text: str, _ms: int) -> None:
        self._status.setText(text)


# ─────────────────────────────────────────────────────────────────
#  Formatting
# ─────────────────────────────────────────────────────────────────


def _describe_roi(roi: ROI) -> str:
    return f"{type(roi.shape).__name__.lower()} ({roi.operation})"


def _format_list_row(index: int, roi: ROI) -> str:
    glyph = "+" if roi.operation == "add" else "−"
    label = f' "{roi.label}"' if roi.label else ""
    return f"{glyph}  #{index}: {type(roi.shape).__name__}{label}"


__all__ = ["RoiEditDialog", "RoiEditPanel"]

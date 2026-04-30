# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — roi.main_window                                        ║
# ║  « ROI builder QMainWindow with shape tools + list panel »       ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Toolbar holds the three shape tools and the mode toggle. Right  ║
# ║  panel is a list of existing ROIs with a per-row op toggle and   ║
# ║  delete; selecting an item highlights it on the canvas. Save /   ║
# ║  Save as / Load mirror the annotator and tuner conventions so    ║
# ║  the user has the same muscle memory across the suite.           ║
# ╚══════════════════════════════════════════════════════════════════╝
"""ROI builder main window."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from pylace.annotator.sidecar import (
    Sidecar,
    default_sidecar_path,
    read_sidecar,
)
from pylace.roi.canvas import RoiCanvas, ToolName
from pylace.roi.geometry import ROI, ROIMode, ROIOperation, ROISet
from pylace.roi.sidecar import (
    ROISidecar,
    ROISidecarSchemaError,
    default_rois_path,
    read_rois,
    write_rois,
)


class RoiBuilderWindow(QtWidgets.QMainWindow):
    """Top-level window for ``pylace-roi``."""

    def __init__(
        self,
        video: Path,
        rois_path: Path,
        arena_sidecar: Sidecar | None,
    ) -> None:
        super().__init__()
        self._video = video
        self._rois_path = rois_path
        self._arena_sidecar = arena_sidecar
        self._roi_set: ROISet = ROISet()

        self._canvas = RoiCanvas(self)
        self._canvas.roiAdded.connect(self._on_roi_added)
        self._canvas.set_arena(arena_sidecar.arena if arena_sidecar else None)
        self._load_first_frame()

        self.setWindowTitle(f"pylace-roi — {video.name}")
        self.setCentralWidget(self._build_central())

        self._build_toolbar()
        self.statusBar().showMessage("Ready.")

        self._load_existing_sidecar()

    # ── Setup ──────────────────────────────────────────────────────────

    def _load_first_frame(self) -> None:
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

    def _build_central(self) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget(self)
        h = QtWidgets.QHBoxLayout(wrap)
        scroll = QtWidgets.QScrollArea(wrap)
        scroll.setWidget(self._canvas)
        scroll.setWidgetResizable(False)
        h.addWidget(scroll, stretch=1)
        h.addWidget(self._build_right_panel(wrap))
        return wrap

    def _build_right_panel(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(parent)
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)
        panel.setMinimumWidth(280)

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

    def _build_mode_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Mode", parent)
        v = QtWidgets.QVBoxLayout(box)
        self._rb_merge = QtWidgets.QRadioButton("Merge (single combined mask)", box)
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

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("Tools")
        bar.setMovable(False)

        circle = QtGui.QAction("Circle", self)
        circle.setShortcut(QtGui.QKeySequence("C"))
        circle.triggered.connect(lambda: self._canvas.set_tool("circle"))
        bar.addAction(circle)

        rect = QtGui.QAction("Rectangle", self)
        rect.setShortcut(QtGui.QKeySequence("R"))
        rect.triggered.connect(lambda: self._canvas.set_tool("rectangle"))
        bar.addAction(rect)

        poly = QtGui.QAction("Polygon", self)
        poly.setShortcut(QtGui.QKeySequence("P"))
        poly.triggered.connect(lambda: self._canvas.set_tool("polygon"))
        bar.addAction(poly)

        bar.addSeparator()

        clear_tool = QtGui.QAction("No tool", self)
        clear_tool.setShortcut(QtGui.QKeySequence("Esc"))
        clear_tool.triggered.connect(lambda: self._canvas.set_tool(None))
        bar.addAction(clear_tool)

        bar.addSeparator()

        quit_act = QtGui.QAction("Quit", self)
        quit_act.setShortcut(QtGui.QKeySequence("Ctrl+Q"))
        quit_act.triggered.connect(self.close)
        bar.addAction(quit_act)

    # ── Slots ─────────────────────────────────────────────────────────

    def _on_roi_added(self, roi: ROI) -> None:
        self._roi_set.add(roi)
        self._refresh_list()
        self._canvas.set_roi_set(self._roi_set)
        self.statusBar().showMessage(
            f"Added {_describe_roi(roi)} (#{len(self._roi_set.rois) - 1}).", 4000,
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

    def _on_delete_selected(self) -> None:
        index = self._list.currentRow()
        if index < 0:
            return
        self._roi_set.remove_at(index)
        self._refresh_list()
        self._canvas.set_roi_set(self._roi_set)

    def _on_save(self) -> None:
        self._write_to(self._rois_path)
        self.statusBar().showMessage(f"Saved to {self._rois_path.name}", 5000)

    def _on_save_as(self) -> None:
        path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save ROI preset",
            str(self._rois_path.parent / "preset.pylace_rois.json"),
            "pyLACE ROIs (*.json);;All files (*)",
        )
        if not path_str:
            return
        self._write_to(Path(path_str))
        self.statusBar().showMessage(f"Saved to {Path(path_str).name}", 5000)

    def _on_load(self) -> None:
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load ROI preset", str(self._rois_path.parent),
            "pyLACE ROIs (*.json);;All files (*)",
        )
        if not path_str:
            return
        try:
            sidecar = read_rois(Path(path_str))
        except (ROISidecarSchemaError, ValueError, OSError) as exc:
            self.statusBar().showMessage(f"Cannot load: {exc}", 7000)
            return
        self._roi_set = sidecar.roi_set
        self._refresh_list()
        self._canvas.set_roi_set(self._roi_set)
        self._sync_mode_radios()
        self.statusBar().showMessage(f"Loaded {Path(path_str).name}", 5000)

    def _set_mode(self, mode: ROIMode) -> None:
        self._roi_set.mode = mode

    # ── Internal ───────────────────────────────────────────────────────

    def _load_existing_sidecar(self) -> None:
        if not self._rois_path.exists():
            return
        try:
            sidecar = read_rois(self._rois_path)
        except ROISidecarSchemaError as exc:
            self.statusBar().showMessage(f"Existing ROI sidecar ignored: {exc}", 7000)
            return
        self._roi_set = sidecar.roi_set
        self._refresh_list()
        self._canvas.set_roi_set(self._roi_set)
        self._sync_mode_radios()

    def _refresh_list(self) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for index, roi in enumerate(self._roi_set.rois):
            self._list.addItem(_format_list_row(index, roi))
        self._list.blockSignals(False)
        if self._roi_set.rois:
            self._list.setCurrentRow(min(self._list.count() - 1, max(0, self._list.currentRow())))

    def _sync_mode_radios(self) -> None:
        if self._roi_set.mode == "split":
            self._rb_split.setChecked(True)
        else:
            self._rb_merge.setChecked(True)

    def _write_to(self, path: Path) -> None:
        sha = (
            self._arena_sidecar.video.sha256 if self._arena_sidecar else ""
        )
        sidecar = ROISidecar(
            video_path=str(self._video),
            video_sha256=sha,
            roi_set=self._roi_set,
        )
        write_rois(sidecar, path)


# ─────────────────────────────────────────────────────────────────
#  Formatting helpers
# ─────────────────────────────────────────────────────────────────

def _describe_roi(roi: ROI) -> str:
    return f"{type(roi.shape).__name__.lower()} ({roi.operation})"


def _format_list_row(index: int, roi: ROI) -> str:
    glyph = "+" if roi.operation == "add" else "−"
    label = f' "{roi.label}"' if roi.label else ""
    return f"{glyph}  #{index}: {type(roi.shape).__name__}{label}"


def run(video: Path, rois_path: Path | None = None) -> int:
    """Launch the GUI event loop. Returns a process exit code."""
    out_path = rois_path if rois_path is not None else default_rois_path(video)
    arena_path = default_sidecar_path(video)
    arena_sidecar: Sidecar | None = None
    if arena_path.exists():
        try:
            arena_sidecar = read_sidecar(arena_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: cannot read arena sidecar: {exc}", file=sys.stderr)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = RoiBuilderWindow(
        video=video, rois_path=out_path, arena_sidecar=arena_sidecar,
    )
    window.resize(1100, 720)
    window.show()
    return app.exec()


__all__ = ["RoiBuilderWindow", "run"]

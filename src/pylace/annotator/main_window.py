"""PyQt6 main window for ``pylace-annotate``."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from pylace.annotator.geometry import (
    Arena,
    Calibration,
    Circle,
    Polygon,
    Rectangle,
    Vertex,
    WorldFrame,
    edge_length,
)
from pylace.annotator.sidecar import (
    Sidecar,
    SidecarSchemaError,
    VideoMeta,
    default_sidecar_path,
    probe_video,
    read_sidecar,
    video_sha256,
    write_sidecar,
)

SNAP_RADIUS_PX = 12.0
HANDLE_SIZE_PX = 8


class FrameCanvas(QtWidgets.QWidget):
    """Custom widget that displays one video frame and edits one arena shape."""

    statusChanged = QtCore.pyqtSignal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QtGui.QPixmap | None = None
        self._mode: str = "idle"
        self._arena: Arena | None = None
        self._world_frame: WorldFrame | None = None
        self._calibration: Calibration | None = None
        self._polygon_in_progress: list[Vertex] = []
        self._draft_origin_pixel: tuple[float, float] | None = None
        self._calib_v1: int | None = None
        self._cursor_image: tuple[float, float] | None = None
        self._drag_kind: str | None = None
        self._drag_index: int | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ── Public state accessors ─────────────────────────────────────────

    def arena(self) -> Arena | None:
        return self._arena

    def world_frame(self) -> WorldFrame | None:
        return self._world_frame

    def calibration(self) -> Calibration | None:
        return self._calibration

    def set_frame(self, frame_rgb: np.ndarray) -> None:
        h, w, _ = frame_rgb.shape
        image = QtGui.QImage(
            frame_rgb.tobytes(), w, h, w * 3,
            QtGui.QImage.Format.Format_RGB888,
        )
        self._pixmap = QtGui.QPixmap.fromImage(image)
        self.setFixedSize(w, h)
        self.update()

    def set_arena(self, arena: Arena | None) -> None:
        self._arena = arena
        self._emit_status()
        self.update()

    def set_world_frame(self, frame: WorldFrame | None) -> None:
        self._world_frame = frame
        self._emit_status()
        self.update()

    def set_calibration(self, cal: Calibration | None) -> None:
        self._calibration = cal
        self._emit_status()
        self.update()

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        if mode != "draw_polygon":
            self._polygon_in_progress.clear()
        if not mode.startswith("pick_calib"):
            self._calib_v1 = None
        self._drag_kind = None
        self._drag_index = None
        self._emit_status()
        self.update()

    # ── Painting ────────────────────────────────────────────────────────

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        if self._pixmap is not None:
            painter.drawPixmap(0, 0, self._pixmap)
        if self._arena is not None:
            self._paint_arena(painter, self._arena)
        elif self._mode == "draw_polygon" and self._polygon_in_progress:
            self._paint_polygon_in_progress(painter)
        if self._mode == "edit":
            self._paint_handles(painter)
        if self._mode == "pick_origin" and self._arena is not None:
            self._paint_snap_ghosts(painter, self._arena.origin_candidates())
        if self._mode in ("pick_calib_v1", "pick_calib_v2"):
            self._paint_calib_picker(painter)
        if self._world_frame is not None:
            self._paint_origin_marker(painter, self._world_frame.origin_pixel)
        if self._calibration is not None and self._arena is not None:
            self._paint_calibration_edge(painter)

    def _paint_arena(self, p: QtGui.QPainter, arena: Arena) -> None:
        pen = QtGui.QPen(QtGui.QColor(0, 220, 220), 2)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if isinstance(arena, Circle):
            p.drawEllipse(
                QtCore.QPointF(arena.cx, arena.cy), arena.r, arena.r,
            )
        else:
            poly = QtGui.QPolygonF(
                [QtCore.QPointF(x, y) for x, y in arena.vertices]
            )
            p.drawPolygon(poly)

    def _paint_polygon_in_progress(self, p: QtGui.QPainter) -> None:
        pen = QtGui.QPen(QtGui.QColor(0, 220, 220), 2, Qt.PenStyle.DashLine)
        p.setPen(pen)
        verts = self._polygon_in_progress
        for i in range(len(verts) - 1):
            p.drawLine(*self._line(verts[i], verts[i + 1]))
        if self._cursor_image is not None and verts:
            p.drawLine(*self._line(verts[-1], self._cursor_image))
        for v in verts:
            self._draw_handle(p, v, QtGui.QColor(0, 220, 220))

    def _paint_handles(self, p: QtGui.QPainter) -> None:
        if self._arena is None:
            return
        for _label, v in self._arena.origin_candidates():
            self._draw_handle(p, v, QtGui.QColor(255, 255, 255))

    def _paint_snap_ghosts(
        self, p: QtGui.QPainter, points: list[tuple[str, Vertex]],
    ) -> None:
        nearest = self._nearest_point([v for _, v in points])
        for i, (_label, v) in enumerate(points):
            colour = (
                QtGui.QColor(255, 220, 0) if i == nearest
                else QtGui.QColor(160, 160, 160)
            )
            self._draw_handle(p, v, colour)

    def _paint_calib_picker(self, p: QtGui.QPainter) -> None:
        if self._arena is None or isinstance(self._arena, Circle):
            return
        verts = self._arena.vertices
        nearest = self._nearest_point(verts)
        for i, v in enumerate(verts):
            if i == self._calib_v1:
                self._draw_handle(p, v, QtGui.QColor(0, 200, 80))
            elif i == nearest:
                self._draw_handle(p, v, QtGui.QColor(255, 220, 0))
            else:
                self._draw_handle(p, v, QtGui.QColor(160, 160, 160))

    def _paint_origin_marker(self, p: QtGui.QPainter, point: Vertex) -> None:
        x, y = point
        pen = QtGui.QPen(QtGui.QColor(255, 80, 80), 2)
        p.setPen(pen)
        p.drawLine(int(x - 8), int(y), int(x + 8), int(y))
        p.drawLine(int(x), int(y - 8), int(x), int(y + 8))

    def _paint_calibration_edge(self, p: QtGui.QPainter) -> None:
        cal = self._calibration
        arena = self._arena
        if cal is None or arena is None:
            return
        if cal.reference_kind == "edge" and cal.reference_vertices is not None:
            i, j = cal.reference_vertices
            verts = arena.vertices  # type: ignore[union-attr]
            pen = QtGui.QPen(QtGui.QColor(255, 80, 220), 3)
            p.setPen(pen)
            p.drawLine(*self._line(verts[i], verts[j]))

    @staticmethod
    def _line(a: Vertex, b: Vertex) -> tuple[int, int, int, int]:
        return int(a[0]), int(a[1]), int(b[0]), int(b[1])

    @staticmethod
    def _draw_handle(p: QtGui.QPainter, v: Vertex, colour: QtGui.QColor) -> None:
        x, y = v
        rect = QtCore.QRectF(
            x - HANDLE_SIZE_PX / 2, y - HANDLE_SIZE_PX / 2,
            HANDLE_SIZE_PX, HANDLE_SIZE_PX,
        )
        p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1))
        p.setBrush(QtGui.QBrush(colour))
        p.drawRect(rect)

    # ── Mouse handling ─────────────────────────────────────────────────

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        p = self._image_coords(event.position())
        button = event.button()
        if self._mode == "draw_circle" and button == Qt.MouseButton.LeftButton:
            self._arena = Circle(p[0], p[1], 1.0)
            self._mode = "drag_circle"
        elif (
            self._mode == "draw_rectangle"
            and button == Qt.MouseButton.LeftButton
        ):
            self._arena = Rectangle.from_two_points(p, (p[0] + 1.0, p[1] + 1.0))
            self._mode = "drag_rectangle"
            self._drag_index = 2  # opposite corner; canonical order
        elif self._mode == "draw_polygon":
            if button == Qt.MouseButton.LeftButton:
                self._polygon_in_progress.append(p)
            elif (
                button == Qt.MouseButton.RightButton
                and len(self._polygon_in_progress) >= 3
            ):
                self._arena = Polygon(self._polygon_in_progress.copy())
                self.set_mode("edit")
                return
        elif self._mode == "edit" and button == Qt.MouseButton.LeftButton:
            self._maybe_grab_handle(p)
        elif self._mode == "pick_origin" and button == Qt.MouseButton.LeftButton:
            self._commit_origin_pick(p)
        elif (
            self._mode == "pick_calib_v1"
            and button == Qt.MouseButton.LeftButton
        ):
            self._commit_calib_v1(p)
        elif (
            self._mode == "pick_calib_v2"
            and button == Qt.MouseButton.LeftButton
        ):
            self._commit_calib_v2(p)
        self._emit_status()
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        self._cursor_image = self._image_coords(event.position())
        if self._mode == "drag_circle" and isinstance(self._arena, Circle):
            c = self._arena
            r = max(1.0, math.hypot(self._cursor_image[0] - c.cx, self._cursor_image[1] - c.cy))
            self._arena = Circle(c.cx, c.cy, r)
        elif self._mode == "drag_rectangle" and isinstance(self._arena, Rectangle):
            opp = self._arena.vertices[(self._drag_index + 2) % 4]
            self._arena = Rectangle.from_two_points(self._cursor_image, opp)
            self._drag_index = self._index_of_nearest(self._arena.vertices, self._cursor_image)
        elif self._mode == "drag_handle":
            self._update_dragged_handle(self._cursor_image)
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._mode in ("drag_circle", "drag_rectangle"):
            self.set_mode("edit")
        elif self._mode == "drag_handle":
            self.set_mode("edit")

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Backspace and self._mode == "draw_polygon":
            if self._polygon_in_progress:
                self._polygon_in_progress.pop()
                self.update()
        elif event.key() == Qt.Key.Key_Escape:
            self._polygon_in_progress.clear()
            self.set_mode("idle" if self._arena is None else "edit")

    # ── Internal helpers ───────────────────────────────────────────────

    def _image_coords(self, pos: QtCore.QPointF) -> tuple[float, float]:
        return (pos.x(), pos.y())

    def _maybe_grab_handle(self, p: Vertex) -> None:
        if self._arena is None:
            return
        if isinstance(self._arena, Circle):
            candidates = self._arena.origin_candidates()
            i = self._index_of_nearest([v for _, v in candidates], p)
            if i is None:
                return
            # Pull the label directly from origin_candidates so the
            # drag_kind always matches the handle the user clicked,
            # regardless of the order Circle uses for its candidate list.
            self._drag_kind = candidates[i][0]
            self._drag_index = i
            self._mode = "drag_handle"
            return
        i = self._index_of_nearest(self._arena.vertices, p)
        if i is None:
            return
        self._drag_kind = "vertex"
        self._drag_index = i
        self._mode = "drag_handle"

    def _update_dragged_handle(self, p: Vertex) -> None:
        if self._arena is None or self._drag_index is None:
            return
        if isinstance(self._arena, Circle):
            cx, cy, r = self._arena.cx, self._arena.cy, self._arena.r
            if self._drag_kind == "centre":
                self._arena = Circle(p[0], p[1], r)
            else:
                new_r = max(1.0, math.hypot(p[0] - cx, p[1] - cy))
                self._arena = Circle(cx, cy, new_r)
            return
        verts = list(self._arena.vertices)
        verts[self._drag_index] = p
        if isinstance(self._arena, Rectangle):
            self._arena = Rectangle(verts)
        else:
            self._arena = Polygon(verts)

    def _commit_origin_pick(self, cursor: Vertex) -> None:
        if self._arena is None:
            return
        candidates = self._arena.origin_candidates()
        nearest = self._nearest_point([v for _, v in candidates])
        if nearest is None:
            return
        _label, point = candidates[nearest]
        existing = self._world_frame
        self._world_frame = WorldFrame(
            origin_pixel=point,
            y_axis=existing.y_axis if existing else "up",
            x_axis=existing.x_axis if existing else "right",
        )
        self.set_mode("edit")

    def _commit_calib_v1(self, cursor: Vertex) -> None:
        if self._arena is None or isinstance(self._arena, Circle):
            return
        nearest = self._nearest_point(self._arena.vertices)
        if nearest is None:
            return
        self._calib_v1 = nearest
        self._mode = "pick_calib_v2"

    def _commit_calib_v2(self, cursor: Vertex) -> None:
        if (
            self._arena is None
            or isinstance(self._arena, Circle)
            or self._calib_v1 is None
        ):
            return
        nearest = self._nearest_point(self._arena.vertices)
        if nearest is None or nearest == self._calib_v1:
            return
        i, j = self._calib_v1, nearest
        pixel = edge_length(self._arena.vertices, i, j)
        mm = self._prompt_mm(f"Edge length (mm) between vertices {i} and {j}:")
        if mm is None:
            self.set_mode("edit")
            return
        self._calibration = Calibration(
            reference_kind="edge",
            physical_mm=mm,
            pixel_distance=pixel,
            reference_vertices=(i, j),
        )
        self.set_mode("edit")

    def _prompt_mm(self, prompt: str) -> float | None:
        value, ok = QtWidgets.QInputDialog.getDouble(
            self, "Calibration", prompt,
            value=10.0, min=0.001, max=100000.0, decimals=3,
        )
        return float(value) if ok else None

    def _nearest_point(self, points: list[Vertex]) -> int | None:
        if self._cursor_image is None or not points:
            return None
        cx, cy = self._cursor_image
        best_i, best_d = None, math.inf
        for i, (x, y) in enumerate(points):
            d = math.hypot(x - cx, y - cy)
            if d < best_d:
                best_i, best_d = i, d
        return best_i

    @staticmethod
    def _index_of_nearest(points: list[Vertex], target: Vertex) -> int | None:
        if not points:
            return None
        tx, ty = target
        best_i, best_d = None, math.inf
        for i, (x, y) in enumerate(points):
            d = math.hypot(x - tx, y - ty)
            if d < best_d and d <= SNAP_RADIUS_PX:
                best_i, best_d = i, d
        return best_i

    def _emit_status(self) -> None:
        self.statusChanged.emit(self._status_text())

    def _status_text(self) -> str:
        prompts = {
            "idle": "Pick a tool from the toolbar.",
            "draw_circle": "Click and drag to draw a circle.",
            "drag_circle": "Drag to set the radius; release to finish.",
            "draw_rectangle": "Click and drag opposite corners to draw a rectangle.",
            "drag_rectangle": "Drag to set the size; release to finish.",
            "draw_polygon": (
                "Left-click to place vertices. Right-click to close. "
                "Backspace removes the last vertex."
            ),
            "edit": "Drag handles to refine, or pick another tool.",
            "drag_handle": "Drag to move the handle.",
            "pick_origin": "Click near a real point on the shape to set the origin.",
            "pick_calib_v1": "Click the first vertex of the calibration edge.",
            "pick_calib_v2": "Click the second vertex of the calibration edge.",
        }
        text = prompts.get(self._mode, self._mode)
        if self._calibration is not None:
            text += f"  |  {self._calibration.mm_per_pixel:.4f} mm/px"
        return text


class AnnotatorWindow(QtWidgets.QMainWindow):
    """Top-level window for ``pylace-annotate``."""

    def __init__(self, video: Path, frame_index: int, out_path: Path) -> None:
        super().__init__()
        self._video = video
        self._out_path = out_path
        self._frame_index = frame_index
        self._video_meta = self._load_video_meta()

        from pylace.annotator.sidecar import Trim
        self._trim = Trim()
        self._bg_max: np.ndarray | None = None
        self._bg_min: np.ndarray | None = None
        self._view_mode: str = "frame"  # "frame" | "bg_max" | "bg_min"

        self.canvas = FrameCanvas(self)
        self.canvas.statusChanged.connect(self._on_status)
        self.canvas.statusChanged.connect(
            lambda _: self._update_step_indicators(),
        )
        self._first_frame_rgb = self._load_first_frame()

        self.setWindowTitle(f"pylace-annotate — {video.name}")
        self.setCentralWidget(self._build_central())
        self.statusBar().showMessage("Ready.")

        self._build_toolbar()
        self._load_existing_sidecar()
        self._maybe_load_cached_bg_pair()
        # Snapshot of the on-disk state (matches _current_state() if a
        # sidecar was loaded, all-None otherwise). closeEvent uses this
        # to decide whether unsaved changes warrant a warning.
        self._saved_state = self._current_state()
        self._sync_view_widgets_from_state()
        self._update_step_indicators()

    def _load_video_meta(self) -> VideoMeta:
        size, fps = probe_video(self._video)
        return VideoMeta(
            path=str(self._video),
            sha256=video_sha256(self._video),
            frame_size=size,
            fps=fps,
        )

    def _load_first_frame(self) -> np.ndarray:
        import cv2

        cap = cv2.VideoCapture(str(self._video))
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {self._video}")
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(self._frame_index))
            ok, frame_bgr = cap.read()
            if not ok:
                raise OSError(f"Cannot read frame {self._frame_index} of {self._video}.")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
        self.canvas.set_frame(frame_rgb)
        return frame_rgb

    def _build_central(self) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget(self)
        v = QtWidgets.QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.canvas, stretch=1)
        v.addWidget(self._build_view_panel(wrap))
        v.addWidget(self._build_nav_strip(wrap))
        return wrap

    def _build_nav_strip(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        from pylace.widgets.navigation import FrameNavigationStrip
        total = self._probe_total_frames()
        self._nav = FrameNavigationStrip(total, parent=parent)
        self._nav.set_fps(self._video_meta.fps)
        self._nav.set_current_frame(self._frame_index)
        self._nav.currentFrameChanged.connect(self._on_nav_frame_changed)
        # The strip's internal range-delimiter is the trim semantic for
        # us; wire its rangeChanged to update _trim.
        self._nav._range_bar.rangeChanged.connect(self._on_nav_range_changed)
        return self._nav

    def _probe_total_frames(self) -> int:
        import cv2

        cap = cv2.VideoCapture(str(self._video))
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        return max(1, n)

    def _build_view_panel(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget(parent)
        h = QtWidgets.QHBoxLayout(wrap)
        h.setContentsMargins(6, 4, 6, 4)

        h.addWidget(QtWidgets.QLabel("View:", wrap))
        self._cb_view = QtWidgets.QComboBox(wrap)
        self._cb_view.addItem("Sample frame", "frame")
        self._cb_view.addItem("Background max", "bg_max")
        self._cb_view.addItem("Background min", "bg_min")
        self._cb_view.setToolTip(
            "Draw the arena on a single sample frame OR on a max/min "
            "background projection over the trim range. The projection "
            "averages out a slightly-misaligned start so the arena "
            "boundary you draw is robust to a wobbly first second.",
        )
        self._cb_view.currentIndexChanged.connect(self._on_view_changed)
        h.addWidget(self._cb_view)

        h.addSpacing(12)
        h.addWidget(QtWidgets.QLabel("Trim start (s):", wrap))
        self._sb_trim_start = QtWidgets.QDoubleSpinBox(wrap)
        self._sb_trim_start.setRange(0.0, 1e9)
        self._sb_trim_start.setDecimals(2)
        self._sb_trim_start.setSpecialValueText("auto")
        self._sb_trim_start.setToolTip(
            "Start of the useful interval in seconds. Saved into the "
            "arena sidecar and read by pylace-detect as the default "
            "--start. Leave at 0 (auto) for whole-video detection.",
        )
        self._sb_trim_start.valueChanged.connect(self._on_trim_changed)
        h.addWidget(self._sb_trim_start)

        h.addWidget(QtWidgets.QLabel("end (s):", wrap))
        self._sb_trim_end = QtWidgets.QDoubleSpinBox(wrap)
        self._sb_trim_end.setRange(0.0, 1e9)
        self._sb_trim_end.setDecimals(2)
        self._sb_trim_end.setSpecialValueText("auto")
        self._sb_trim_end.setToolTip(
            "End of the useful interval in seconds. Saved into the "
            "arena sidecar and read by pylace-detect as the default "
            "--end. Leave at 0 (auto) for whole-video detection.",
        )
        self._sb_trim_end.valueChanged.connect(self._on_trim_changed)
        h.addWidget(self._sb_trim_end)

        set_start = QtWidgets.QPushButton("Set start = current", wrap)
        set_start.setToolTip(
            "Take the scrubber's current frame as the trim start.",
        )
        set_start.clicked.connect(self._on_set_trim_start_to_current)
        h.addWidget(set_start)

        set_end = QtWidgets.QPushButton("Set end = current", wrap)
        set_end.setToolTip(
            "Take the scrubber's current frame as the trim end.",
        )
        set_end.clicked.connect(self._on_set_trim_end_to_current)
        h.addWidget(set_end)

        h.addSpacing(8)
        recompute = QtWidgets.QPushButton("Recompute background", wrap)
        recompute.setToolTip(
            "Sample the current trim range and rebuild the max/min "
            "projection pair. Caches to <video>.pylace_background_*.png.",
        )
        recompute.clicked.connect(self._on_recompute_bg)
        h.addWidget(recompute)

        h.addStretch(1)
        return wrap

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("Tools")
        toolbar.setMovable(False)

        self._arena_status = self._add_step_indicator(toolbar, "Arena")
        self._circle_action = self._add_mode_action(toolbar, "Circle", "draw_circle")
        self._rect_action = self._add_mode_action(toolbar, "Rectangle", "draw_rectangle")
        self._polygon_action = self._add_mode_action(toolbar, "Polygon", "draw_polygon")
        toolbar.addSeparator()

        self._origin_status = self._add_step_indicator(toolbar, "Origin")
        origin_action = QtGui.QAction("Set origin", self)
        origin_action.triggered.connect(self._start_origin_pick)
        toolbar.addAction(origin_action)

        self._calibrate_status = self._add_step_indicator(toolbar, "Calibrate")
        calibrate_action = QtGui.QAction("Set calibration", self)
        calibrate_action.triggered.connect(self._start_calibration)
        toolbar.addAction(calibrate_action)

        toolbar.addSeparator()

        self._y_up_action = QtGui.QAction("Y up", self)
        self._y_up_action.setCheckable(True)
        self._y_up_action.setChecked(True)
        self._y_up_action.toggled.connect(self._on_y_up_toggled)
        toolbar.addAction(self._y_up_action)

        toolbar.addSeparator()

        save_action = QtGui.QAction("Save", self)
        save_action.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self._save)
        toolbar.addAction(save_action)

        quit_action = QtGui.QAction("Quit", self)
        quit_action.setShortcut(QtGui.QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        toolbar.addAction(quit_action)

    def _add_mode_action(
        self, toolbar: QtWidgets.QToolBar, label: str, mode: str,
    ) -> QtGui.QAction:
        action = QtGui.QAction(label, self)
        action.triggered.connect(lambda: self.canvas.set_mode(mode))
        toolbar.addAction(action)
        return action

    def _add_step_indicator(
        self, toolbar: QtWidgets.QToolBar, name: str,
    ) -> QtWidgets.QLabel:
        """Toolbar status pill for one of the three required steps."""
        label = QtWidgets.QLabel(f"✗ {name}", self)
        label.setStyleSheet(
            "QLabel {"
            "  color: #c0392b;"
            "  font-weight: bold;"
            "  padding: 2px 8px;"
            "  margin-right: 2px;"
            "}"
        )
        label.setToolTip(f"{name}: not set yet")
        action = QtWidgets.QWidgetAction(self)
        action.setDefaultWidget(label)
        toolbar.addAction(action)
        return label

    def _set_step_indicator(self, label: QtWidgets.QLabel, name: str, done: bool) -> None:
        if done:
            label.setText(f"✓ {name}")
            label.setStyleSheet(
                "QLabel {"
                "  color: #1e8449;"
                "  font-weight: bold;"
                "  padding: 2px 8px;"
                "  margin-right: 2px;"
                "}"
            )
            label.setToolTip(f"{name}: complete")
        else:
            label.setText(f"✗ {name}")
            label.setStyleSheet(
                "QLabel {"
                "  color: #c0392b;"
                "  font-weight: bold;"
                "  padding: 2px 8px;"
                "  margin-right: 2px;"
                "}"
            )
            label.setToolTip(f"{name}: not set yet")

    def _update_step_indicators(self) -> None:
        self._set_step_indicator(
            self._arena_status, "Arena",
            self.canvas.arena() is not None,
        )
        self._set_step_indicator(
            self._origin_status, "Origin",
            self.canvas.world_frame() is not None,
        )
        self._set_step_indicator(
            self._calibrate_status, "Calibrate",
            self.canvas.calibration() is not None,
        )

    def _missing_steps(self) -> list[str]:
        missing = []
        if self.canvas.arena() is None:
            missing.append("Arena (draw a Circle / Rectangle / Polygon on the frame)")
        if self.canvas.world_frame() is None:
            missing.append("Origin (click Set origin then click a point on the arena)")
        if self.canvas.calibration() is None:
            missing.append(
                "Calibration (click Set calibration; for a circle the diameter "
                "in mm; for polygons click two points whose real distance you know)",
            )
        return missing

    def _current_state(self):
        return (
            self.canvas.arena(),
            self.canvas.world_frame(),
            self.canvas.calibration(),
            (self._trim.start_s, self._trim.end_s),
        )

    # ── Trim + view-mode plumbing ──────────────────────────────────────

    def _sync_view_widgets_from_state(self) -> None:
        """Push the current _trim values back into the spinboxes."""
        if not hasattr(self, "_sb_trim_start"):
            return
        self._sb_trim_start.blockSignals(True)
        self._sb_trim_start.setValue(self._trim.start_s or 0.0)
        self._sb_trim_start.blockSignals(False)
        self._sb_trim_end.blockSignals(True)
        self._sb_trim_end.setValue(self._trim.end_s or 0.0)
        self._sb_trim_end.blockSignals(False)
        # Disable bg-view options if no projection is loaded yet.
        bg_ready = self._bg_max is not None and self._bg_min is not None
        for i in (1, 2):
            self._cb_view.model().item(i).setEnabled(bg_ready)
        if not bg_ready:
            self._cb_view.setToolTip(
                "Click Recompute background to sample the trim range and "
                "build the max/min projection pair.",
            )

    def _on_view_changed(self, _index: int) -> None:
        mode = self._cb_view.currentData() or "frame"
        if mode in ("bg_max", "bg_min") and (
            self._bg_max is None or self._bg_min is None
        ):
            self.statusBar().showMessage(
                "No background pair on disk. Click Recompute background.",
                4000,
            )
            self._cb_view.blockSignals(True)
            self._cb_view.setCurrentIndex(0)
            self._cb_view.blockSignals(False)
            return
        self._view_mode = mode
        self._apply_view_mode()

    def _on_nav_frame_changed(self, frame: int) -> None:
        """Strip scrubber moved → load that frame onto the canvas."""
        self._frame_index = int(frame)
        if self._view_mode == "frame":
            self._first_frame_rgb = self._load_first_frame()

    def _on_nav_range_changed(self, lo: int, hi: int) -> None:
        """Strip's yellow handles moved → push into _trim + spinboxes."""
        from pylace.annotator.sidecar import Trim
        fps = max(1.0, float(self._video_meta.fps))
        total = self._probe_total_frames()
        # The strip always carves a non-trivial sub-range out of the
        # total movie span. Treat lo == 0 as "no start trim" and
        # hi == total - 1 as "no end trim", so dragging the handles
        # in to the very edges clears the corresponding bound.
        start_s = (lo / fps) if lo > 0 else None
        end_s = (hi / fps) if hi < total - 1 else None
        self._trim = Trim(start_s=start_s, end_s=end_s)
        self._sync_view_widgets_from_state()

    def _on_trim_changed(self, _value: float) -> None:
        """Spinbox edited → push into _trim + nav strip's range bar."""
        from pylace.annotator.sidecar import Trim
        s = float(self._sb_trim_start.value())
        e = float(self._sb_trim_end.value())
        self._trim = Trim(
            start_s=s if s > 0.0 else None,
            end_s=e if e > 0.0 else None,
        )
        self._sync_nav_range_from_trim()
        self._on_status(self.canvas._status_text())  # refresh status bar

    def _sync_nav_range_from_trim(self) -> None:
        """Push current ``_trim`` into the nav strip's range delimiter."""
        if not hasattr(self, "_nav"):
            return
        fps = max(1.0, float(self._video_meta.fps))
        total = self._probe_total_frames()
        lo = int(round((self._trim.start_s or 0.0) * fps))
        hi = int(round((self._trim.end_s or (total / fps)) * fps))
        lo = max(0, min(total - 2, lo))
        hi = max(lo + 1, min(total - 1, hi))
        bar = self._nav._range_bar
        bar.blockSignals(True)
        bar._lo = lo
        bar._hi = hi
        bar.update()
        bar.blockSignals(False)
        # Reset the scrubber's range to the new sub-window so scrubbing
        # naturally stays inside the trim.
        self._nav._scrub.blockSignals(True)
        self._nav._scrub.setRange(lo, hi)
        cur = max(lo, min(hi, self._nav.current_frame()))
        self._nav._scrub.setValue(cur)
        self._nav._scrub.blockSignals(False)

    def _on_set_trim_start_to_current(self) -> None:
        """Snap trim start to whatever frame the scrubber is on."""
        from pylace.annotator.sidecar import Trim
        fps = max(1.0, float(self._video_meta.fps))
        cur_s = float(self._frame_index) / fps
        self._trim = Trim(start_s=cur_s, end_s=self._trim.end_s)
        self._sync_view_widgets_from_state()
        self._sync_nav_range_from_trim()

    def _on_set_trim_end_to_current(self) -> None:
        """Snap trim end to whatever frame the scrubber is on."""
        from pylace.annotator.sidecar import Trim
        fps = max(1.0, float(self._video_meta.fps))
        cur_s = float(self._frame_index) / fps
        self._trim = Trim(start_s=self._trim.start_s, end_s=cur_s)
        self._sync_view_widgets_from_state()
        self._sync_nav_range_from_trim()

    def _on_recompute_bg(self) -> None:
        from pylace.detect.background import compute_projection_pair, save_background_png, default_background_paths

        total_s = self._video_meta.frame_size and (
            # n_frames cannot be derived without an extra cv2.VideoCapture;
            # estimate from fps × probed frame count.
            None
        )
        # Convert (start_s, end_s) → (start_frac, end_frac).
        start_frac, end_frac = self._trim_fractions()
        try:
            self.statusBar().showMessage(
                "Computing background pair (this can take a few seconds)…",
            )
            QtWidgets.QApplication.processEvents()
            self._bg_max, self._bg_min = compute_projection_pair(
                self._video, n_frames=50,
                start_frac=start_frac, end_frac=end_frac,
            )
        except (OSError, ValueError) as exc:
            self.statusBar().showMessage(f"Could not compute bg: {exc}", 6000)
            return
        # Cache to disk.
        max_path, min_path = default_background_paths(self._video)
        try:
            save_background_png(self._bg_max, max_path)
            save_background_png(self._bg_min, min_path)
        except OSError as exc:
            self.statusBar().showMessage(
                f"Computed but could not cache: {exc}", 6000,
            )
        self.statusBar().showMessage(
            f"Background pair updated (cached to {max_path.name} / {min_path.name}).",
            5000,
        )
        # Re-enable bg view options.
        for i in (1, 2):
            self._cb_view.model().item(i).setEnabled(True)
        self._apply_view_mode()

    def _maybe_load_cached_bg_pair(self) -> None:
        from pylace.detect.background import default_background_paths, load_background_png
        max_path, min_path = default_background_paths(self._video)
        if not (max_path.exists() and min_path.exists()):
            return
        try:
            self._bg_max = load_background_png(max_path)
            self._bg_min = load_background_png(min_path)
        except OSError:
            self._bg_max = None
            self._bg_min = None

    def _trim_fractions(self) -> tuple[float, float]:
        """Convert ``_trim`` start/end seconds to (start_frac, end_frac)."""
        # Probe total frame count once via cv2 — fps is already known.
        import cv2

        cap = cv2.VideoCapture(str(self._video))
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        if n <= 0 or self._video_meta.fps <= 0:
            return 0.1, 0.9
        duration_s = n / self._video_meta.fps
        s = self._trim.start_s if self._trim.start_s is not None else 0.0
        e = self._trim.end_s if self._trim.end_s is not None else duration_s
        s = max(0.0, min(duration_s - 1e-3, s))
        e = max(s + 1e-3, min(duration_s, e))
        return s / duration_s, e / duration_s

    def _apply_view_mode(self) -> None:
        import cv2
        if self._view_mode == "bg_max" and self._bg_max is not None:
            self.canvas.set_frame(cv2.cvtColor(self._bg_max, cv2.COLOR_GRAY2RGB))
        elif self._view_mode == "bg_min" and self._bg_min is not None:
            self.canvas.set_frame(cv2.cvtColor(self._bg_min, cv2.COLOR_GRAY2RGB))
        else:
            self.canvas.set_frame(self._first_frame_rgb)

    def _start_origin_pick(self) -> None:
        if self.canvas.arena() is None:
            self.statusBar().showMessage("Draw an arena first.", 3000)
            return
        self.canvas.set_mode("pick_origin")

    def _start_calibration(self) -> None:
        arena = self.canvas.arena()
        if arena is None:
            self.statusBar().showMessage("Draw an arena first.", 3000)
            return
        if isinstance(arena, Circle):
            self._prompt_circle_diameter(arena)
        else:
            self.canvas.set_mode("pick_calib_v1")

    def _prompt_circle_diameter(self, circle: Circle) -> None:
        mm, ok = QtWidgets.QInputDialog.getDouble(
            self, "Calibration", "Diameter (mm):",
            value=10.0, min=0.001, max=100000.0, decimals=3,
        )
        if not ok:
            return
        cal = Calibration(
            reference_kind="diameter",
            physical_mm=float(mm),
            pixel_distance=2.0 * circle.r,
        )
        self.canvas.set_calibration(cal)
        # Belt-and-braces refresh in case the canvas's status signal
        # arrives on a different event-loop tick than the dialog return.
        self._update_step_indicators()

    def _on_y_up_toggled(self, checked: bool) -> None:
        existing = self.canvas.world_frame()
        if existing is None:
            return
        self.canvas.set_world_frame(
            WorldFrame(
                origin_pixel=existing.origin_pixel,
                y_axis="up" if checked else "down",
                x_axis=existing.x_axis,
            )
        )

    def _load_existing_sidecar(self) -> None:
        if not self._out_path.exists():
            return
        try:
            sidecar = read_sidecar(self._out_path)
        except SidecarSchemaError as exc:
            self.statusBar().showMessage(f"Existing sidecar ignored: {exc}", 5000)
            return
        if sidecar.video.sha256 != self._video_meta.sha256:
            self.statusBar().showMessage(
                "Existing sidecar SHA256 mismatch; loading geometry but you should re-verify.",
                8000,
            )
        self.canvas.set_arena(sidecar.arena)
        self.canvas.set_world_frame(sidecar.world_frame)
        self.canvas.set_calibration(sidecar.calibration)
        self._y_up_action.setChecked(sidecar.world_frame.y_axis == "up")
        if sidecar.arena is not None:
            self.canvas.set_mode("edit")
        if sidecar.trim is not None:
            self._trim = sidecar.trim
            self._sync_nav_range_from_trim()

    def _save(self) -> bool:
        """Try to write the sidecar; return True on success, False otherwise."""
        missing = self._missing_steps()
        if missing:
            QtWidgets.QMessageBox.warning(
                self,
                "Cannot save annotation",
                "The annotation is incomplete. Please finish:\n\n"
                + "\n".join(f"  •  {step}" for step in missing),
            )
            return False
        sidecar = Sidecar(
            video=self._video_meta,
            arena=self.canvas.arena(),
            world_frame=self.canvas.world_frame(),
            calibration=self.canvas.calibration(),
            trim=self._trim if not self._trim.is_empty() else None,
        )
        write_sidecar(sidecar, self._out_path)
        self._saved_state = self._current_state()
        self.statusBar().showMessage(f"Saved {self._out_path}.", 5000)
        return True

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        if not self._has_unsaved_work():
            super().closeEvent(event)
            return
        missing = self._missing_steps()
        if missing:
            text = (
                "Annotation is incomplete and not yet saved. "
                "Closing now will discard your work.\n\n"
                "Still missing:\n"
                + "\n".join(f"  •  {step}" for step in missing)
                + "\n\nQuit anyway?"
            )
            buttons = (
                QtWidgets.QMessageBox.StandardButton.Discard
                | QtWidgets.QMessageBox.StandardButton.Cancel
            )
            choice = QtWidgets.QMessageBox.question(
                self, "Annotation incomplete", text, buttons,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if choice == QtWidgets.QMessageBox.StandardButton.Discard:
                super().closeEvent(event)
            else:
                event.ignore()
            return
        # All three steps are set but the on-disk file is out of date.
        choice = QtWidgets.QMessageBox.question(
            self, "Save before quitting?",
            "You have unsaved annotation changes. Save now?",
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Save,
        )
        if choice == QtWidgets.QMessageBox.StandardButton.Save:
            if self._save():
                super().closeEvent(event)
            else:
                event.ignore()
        elif choice == QtWidgets.QMessageBox.StandardButton.Discard:
            super().closeEvent(event)
        else:
            event.ignore()

    def _has_unsaved_work(self) -> bool:
        if all(s is None for s in self._current_state()):
            return False  # blank canvas, nothing to lose
        return self._current_state() != self._saved_state

    def _on_status(self, text: str) -> None:
        self.statusBar().showMessage(text)


def run(video: Path, frame_index: int, out: Path | None) -> int:
    """Launch the GUI event loop. Returns a process exit code."""
    out_path = out if out is not None else default_sidecar_path(video)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = AnnotatorWindow(video=video, frame_index=frame_index, out_path=out_path)
    window.show()
    return app.exec()

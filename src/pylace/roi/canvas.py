# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — roi.canvas                                             ║
# ║  « multi-shape ROI canvas with draw-only-then-frozen edit model »║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  v1 limitation: once a shape is added, the canvas treats it as   ║
# ║  immutable — to change a shape, delete it from the list and      ║
# ║  redraw. Per-vertex editing is reserved for a later sprint to    ║
# ║  keep this canvas's selection model trivial (selection only      ║
# ║  drives outline highlighting, not drag handles).                 ║
# ╚══════════════════════════════════════════════════════════════════╝
"""QLabel-based multi-shape canvas for the ROI builder."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from pylace.annotator.geometry import (
    Arena,
    Circle,
    Polygon,
    Rectangle,
    Vertex,
)
from pylace.roi.constants import (
    ARENA_OUTLINE_COLOR_BGR,
    ROI_ADD_COLOR_BGR,
    ROI_DRAFT_COLOR_BGR,
    ROI_FILL_ALPHA,
    ROI_KEPT_FILL_BGR,
    ROI_OUTLINE_PX,
    ROI_SELECTED_COLOR_BGR,
    ROI_SUBTRACT_COLOR_BGR,
)
from pylace.roi.geometry import ROI, ROISet
from pylace.roi.mask import build_combined_mask

ToolName = Literal["circle", "rectangle", "polygon"]


class RoiCanvas(QtWidgets.QLabel):
    """Frame view that lets the user draw multiple ROIs on top."""

    roiAdded = QtCore.pyqtSignal(object)
    """Emitted with the newly-finalised ``ROI`` once a draw stroke ends."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame_rgb: np.ndarray | None = None
        self._roi_set: ROISet = ROISet()
        self._arena: Arena | None = None
        self._tool: ToolName | None = None
        self._selected_index: int = -1
        self._draft_mode: str = "idle"
        self._draft_arena: Arena | None = None
        self._polygon_in_progress: list[Vertex] = []
        self._cursor_pos: tuple[int, int] | None = None
        # Anchor point for the in-progress rectangle drag — fixed at the
        # original click so dragging in any direction grows / shrinks the
        # rectangle relative to the click, not relative to the previous
        # cursor position.
        self._rect_anchor: Vertex | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ── Public API ─────────────────────────────────────────────────────

    def set_frame(self, frame_rgb: np.ndarray) -> None:
        h, w, _ = frame_rgb.shape
        self._frame_rgb = frame_rgb.copy()
        self.setFixedSize(w, h)
        self._refresh()

    def set_arena(self, arena: Arena | None) -> None:
        self._arena = arena
        self._refresh()

    def set_roi_set(self, roi_set: ROISet) -> None:
        self._roi_set = roi_set
        if self._selected_index >= len(roi_set.rois):
            self._selected_index = -1
        self._refresh()

    def set_tool(self, tool: ToolName | None) -> None:
        self._tool = tool
        self._draft_mode = "idle"
        self._draft_arena = None
        self._polygon_in_progress.clear()
        self._refresh()

    def set_selected(self, index: int) -> None:
        if -1 <= index < len(self._roi_set.rois):
            self._selected_index = index
            self._refresh()

    # ── Mouse / key events ─────────────────────────────────────────────

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        pos = self._image_pos(event)
        button = event.button()
        if button == Qt.MouseButton.LeftButton:
            self._on_left_press(pos)
        elif button == Qt.MouseButton.RightButton:
            self._on_right_press(pos)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._cursor_pos = self._image_pos(event)
        if self._draft_mode == "drag_circle":
            self._update_draft_circle()
        elif self._draft_mode == "drag_rectangle":
            self._update_draft_rectangle()
        self._refresh()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._draft_mode in ("drag_circle", "drag_rectangle"):
            self._finalise_draft()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        if (
            event.key() == Qt.Key.Key_Backspace
            and self._draft_mode == "draw_polygon"
        ):
            if self._polygon_in_progress:
                self._polygon_in_progress.pop()
                self._refresh()
        elif event.key() == Qt.Key.Key_Escape:
            self._cancel_draft()

    # ── Internal — draft management ────────────────────────────────────

    def _on_left_press(self, pos: Vertex) -> None:
        if self._tool == "circle":
            self._start_circle_draft(pos)
        elif self._tool == "rectangle":
            self._start_rectangle_draft(pos)
        elif self._tool == "polygon":
            self._extend_polygon_draft(pos)

    def _on_right_press(self, pos: Vertex) -> None:
        if (
            self._tool == "polygon"
            and self._draft_mode == "draw_polygon"
            and len(self._polygon_in_progress) >= 3
        ):
            shape = Polygon(self._polygon_in_progress.copy())
            self._polygon_in_progress.clear()
            self._draft_mode = "idle"
            self.roiAdded.emit(ROI(shape=shape))

    def _start_circle_draft(self, pos: Vertex) -> None:
        self._draft_arena = Circle(pos[0], pos[1], 1.0)
        self._draft_mode = "drag_circle"

    def _start_rectangle_draft(self, pos: Vertex) -> None:
        self._rect_anchor = pos
        self._draft_arena = Rectangle.from_two_points(
            pos, (pos[0] + 1.0, pos[1] + 1.0),
        )
        self._draft_mode = "drag_rectangle"

    def _extend_polygon_draft(self, pos: Vertex) -> None:
        if self._draft_mode != "draw_polygon":
            self._draft_mode = "draw_polygon"
            self._polygon_in_progress = []
        self._polygon_in_progress.append(pos)

    def _update_draft_circle(self) -> None:
        if not isinstance(self._draft_arena, Circle) or self._cursor_pos is None:
            return
        cx, cy = self._draft_arena.cx, self._draft_arena.cy
        radius = max(1.0, ((self._cursor_pos[0] - cx) ** 2 + (self._cursor_pos[1] - cy) ** 2) ** 0.5)
        self._draft_arena = Circle(cx, cy, radius)

    def _update_draft_rectangle(self) -> None:
        if (
            not isinstance(self._draft_arena, Rectangle)
            or self._cursor_pos is None
            or self._rect_anchor is None
        ):
            return
        # Anchor stays at the original click; cursor is the other corner.
        # Rectangle.from_two_points sorts coordinates so any drag direction
        # produces a correctly axis-aligned rectangle.
        self._draft_arena = Rectangle.from_two_points(
            self._rect_anchor, self._cursor_pos,
        )

    def _finalise_draft(self) -> None:
        if self._draft_arena is None:
            return
        if isinstance(self._draft_arena, Circle) and self._draft_arena.r < 2.0:
            self._cancel_draft()
            return
        if isinstance(self._draft_arena, Rectangle):
            verts = self._draft_arena.vertices
            width = abs(verts[1][0] - verts[0][0])
            height = abs(verts[3][1] - verts[0][1])
            if width < 2.0 or height < 2.0:
                self._cancel_draft()
                return
        roi = ROI(shape=self._draft_arena)
        self._draft_arena = None
        self._draft_mode = "idle"
        self._rect_anchor = None
        self.roiAdded.emit(roi)

    def _cancel_draft(self) -> None:
        self._draft_arena = None
        self._draft_mode = "idle"
        self._rect_anchor = None
        self._polygon_in_progress.clear()
        self._refresh()

    # ── Rendering ──────────────────────────────────────────────────────

    @staticmethod
    def _image_pos(event: QtGui.QMouseEvent) -> tuple[int, int]:
        return (int(event.position().x()), int(event.position().y()))

    def _refresh(self) -> None:
        if self._frame_rgb is None:
            return
        bgr = cv2.cvtColor(self._frame_rgb, cv2.COLOR_RGB2BGR)
        self._tint_combined_mask(bgr)
        self._draw_arena(bgr)
        for index, roi in enumerate(self._roi_set.rois):
            self._draw_roi(bgr, roi, selected=index == self._selected_index)
        self._draw_draft(bgr)
        self._publish(bgr)

    def _tint_combined_mask(self, bgr: np.ndarray) -> None:
        """Tint the effective kept region in green so add/subtract math is visible."""
        if not self._roi_set.rois or self._roi_set.mode != "merge":
            return
        height, width = bgr.shape[:2]
        try:
            mask = build_combined_mask(self._roi_set, (width, height))
        except NotImplementedError:
            return
        if not mask.any():
            return
        overlay = bgr.copy()
        overlay[mask] = ROI_KEPT_FILL_BGR
        cv2.addWeighted(
            bgr, 1.0 - ROI_FILL_ALPHA, overlay, ROI_FILL_ALPHA, 0, dst=bgr,
        )

    def _draw_arena(self, bgr: np.ndarray) -> None:
        if self._arena is None:
            return
        _draw_shape(bgr, self._arena, ARENA_OUTLINE_COLOR_BGR, thickness=1)

    def _draw_roi(self, bgr: np.ndarray, roi: ROI, *, selected: bool) -> None:
        if selected:
            colour = ROI_SELECTED_COLOR_BGR
        elif roi.operation == "add":
            colour = ROI_ADD_COLOR_BGR
        else:
            colour = ROI_SUBTRACT_COLOR_BGR
        _draw_shape(bgr, roi.shape, colour, thickness=ROI_OUTLINE_PX)

    def _draw_draft(self, bgr: np.ndarray) -> None:
        if self._draft_arena is not None:
            _draw_shape(
                bgr, self._draft_arena, ROI_DRAFT_COLOR_BGR,
                thickness=ROI_OUTLINE_PX, dashed=True,
            )
        if self._draft_mode == "draw_polygon" and self._polygon_in_progress:
            self._draw_polygon_in_progress(bgr)

    def _draw_polygon_in_progress(self, bgr: np.ndarray) -> None:
        verts = self._polygon_in_progress
        for i in range(len(verts) - 1):
            cv2.line(
                bgr,
                (int(verts[i][0]), int(verts[i][1])),
                (int(verts[i + 1][0]), int(verts[i + 1][1])),
                ROI_DRAFT_COLOR_BGR, ROI_OUTLINE_PX,
            )
        if self._cursor_pos is not None and verts:
            cv2.line(
                bgr,
                (int(verts[-1][0]), int(verts[-1][1])),
                self._cursor_pos,
                ROI_DRAFT_COLOR_BGR, 1,
            )
        for v in verts:
            cv2.circle(
                bgr, (int(v[0]), int(v[1])), 4,
                ROI_DRAFT_COLOR_BGR, thickness=-1,
            )

    def _publish(self, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        image = QtGui.QImage(
            rgb.tobytes(), w, h, w * 3, QtGui.QImage.Format.Format_RGB888,
        )
        self.setPixmap(QtGui.QPixmap.fromImage(image))


# ─────────────────────────────────────────────────────────────────
#  Shape-drawing helpers
# ─────────────────────────────────────────────────────────────────

def _draw_shape(
    bgr: np.ndarray,
    shape: Arena,
    colour: tuple[int, int, int],
    *,
    thickness: int = 1,
    dashed: bool = False,
) -> None:
    if isinstance(shape, Circle):
        _draw_circle(bgr, shape, colour, thickness, dashed)
    else:
        _draw_polygon(bgr, shape.vertices, colour, thickness, dashed)


def _draw_circle(
    bgr: np.ndarray, circle: Circle, colour: tuple[int, int, int],
    thickness: int, dashed: bool,
) -> None:
    centre = (int(round(circle.cx)), int(round(circle.cy)))
    radius = max(1, int(round(circle.r)))
    if not dashed:
        cv2.circle(bgr, centre, radius, colour, thickness)
        return
    # Dashed circle by drawing arc segments.
    for a in range(0, 360, 16):
        cv2.ellipse(bgr, centre, (radius, radius), 0, a, a + 8, colour, thickness)


def _draw_polygon(
    bgr: np.ndarray, vertices: list[Vertex], colour: tuple[int, int, int],
    thickness: int, dashed: bool,
) -> None:
    if len(vertices) < 2:
        return
    pts = np.array(vertices, dtype=np.int32).reshape(-1, 1, 2)
    if not dashed:
        cv2.polylines(bgr, [pts], isClosed=True, color=colour, thickness=thickness)
        return
    n = len(vertices)
    for i in range(n):
        a = (int(vertices[i][0]), int(vertices[i][1]))
        b = (int(vertices[(i + 1) % n][0]), int(vertices[(i + 1) % n][1]))
        _draw_dashed_line(bgr, a, b, colour, thickness)


def _draw_dashed_line(
    bgr: np.ndarray,
    a: tuple[int, int], b: tuple[int, int],
    colour: tuple[int, int, int], thickness: int,
    dash: int = 8,
) -> None:
    ax, ay = a
    bx, by = b
    length = max(1.0, ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5)
    n = int(length // dash)
    for i in range(n):
        if i % 2 == 0:
            t0 = i / max(1, n)
            t1 = (i + 1) / max(1, n)
            p0 = (int(ax + (bx - ax) * t0), int(ay + (by - ay) * t0))
            p1 = (int(ax + (bx - ax) * t1), int(ay + (by - ay) * t1))
            cv2.line(bgr, p0, p1, colour, thickness)


__all__ = ["RoiCanvas", "ToolName"]

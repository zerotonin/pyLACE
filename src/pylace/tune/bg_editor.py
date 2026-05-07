"""Background editor: paint mask + smart fill, clone stamp, healing brush.

Photoshop-equivalent operations on a grayscale background image:

- **Smart fill** — :func:`apply_inpaint`, wrapping ``cv2.inpaint(TELEA)``.
  Paint a mask, click "Inpaint mask", and the algorithm propagates
  surrounding texture inward across the masked region.
- **Clone stamp** — :func:`copy_circle`, direct pixel copy from a
  source point to the cursor. Useful when you can find a clean
  same-luminance patch nearby.
- **Healing brush** — :func:`heal_circle`, ``cv2.seamlessClone``
  Poisson blending. Same UX as clone stamp but blends gradient
  fields with the destination boundary, hiding luminance / colour
  mismatches between source and destination.

The pure helpers are testable without Qt; :class:`BgEditCanvas` and
:class:`BgEditDialog` are the interactive front-end.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

DEFAULT_BRUSH_SIZE = 12
DEFAULT_INPAINT_RADIUS = 3
HISTORY_LIMIT = 32

ToolName = Literal["smart_fill", "clone_stamp", "healing_brush"]
TOOL_HINTS: dict[str, str] = {
    "smart_fill": (
        "Smart fill: paint over the unwanted region with LMB (erase with "
        "RMB), then click 'Apply smart fill' below to fill it from the "
        "surrounding pixels."
    ),
    "clone_stamp": (
        "Clone stamp: Alt-click to set source. Drag with LMB to copy from "
        "source to cursor."
    ),
    "healing_brush": (
        "Healing brush: Alt-click to set source. Drag with LMB; copies "
        "with Poisson blending so luminance matches the surround."
    ),
}


# ── Pure helpers ────────────────────────────────────────────────────────


def apply_inpaint(
    bg: np.ndarray, mask: np.ndarray, radius: int = DEFAULT_INPAINT_RADIUS,
) -> np.ndarray:
    """Fill the masked region of ``bg`` using surrounding pixels (TELEA)."""
    if bg.shape != mask.shape:
        raise ValueError(f"Shape mismatch: bg {bg.shape} vs mask {mask.shape}.")
    if bg.dtype != np.uint8:
        raise ValueError("bg must be uint8.")
    if mask.dtype != np.uint8:
        raise ValueError("mask must be uint8.")
    if radius < 1:
        raise ValueError("radius must be >= 1.")
    return cv2.inpaint(bg, mask, radius, cv2.INPAINT_TELEA)


def copy_circle(
    bg: np.ndarray, *,
    src_x: int, src_y: int, dst_x: int, dst_y: int, radius: int,
) -> None:
    """Direct-copy a circular patch from source to destination, in place.

    Pixels that fall outside ``bg`` on either the source or destination
    side are silently clipped; the in-bounds intersection is the only
    region that gets written.
    """
    if radius < 1:
        raise ValueError("radius must be >= 1.")
    if bg.dtype != np.uint8:
        raise ValueError("bg must be uint8.")
    h, w = bg.shape

    dx0, dy0 = max(0, dst_x - radius), max(0, dst_y - radius)
    dx1 = min(w, dst_x + radius + 1)
    dy1 = min(h, dst_y + radius + 1)
    if dx0 >= dx1 or dy0 >= dy1:
        return

    offset_x = src_x - dst_x
    offset_y = src_y - dst_y
    sx0, sy0 = dx0 + offset_x, dy0 + offset_y
    sx1, sy1 = dx1 + offset_x, dy1 + offset_y
    if sx0 < 0:
        shift = -sx0
        sx0, dx0 = sx0 + shift, dx0 + shift
    if sy0 < 0:
        shift = -sy0
        sy0, dy0 = sy0 + shift, dy0 + shift
    if sx1 > w:
        shift = sx1 - w
        sx1, dx1 = sx1 - shift, dx1 - shift
    if sy1 > h:
        shift = sy1 - h
        sy1, dy1 = sy1 - shift, dy1 - shift
    if dx0 >= dx1 or dy0 >= dy1:
        return

    yy, xx = np.ogrid[dy0:dy1, dx0:dx1]
    mask = (xx - dst_x) ** 2 + (yy - dst_y) ** 2 <= radius ** 2
    src_patch = bg[sy0:sy1, sx0:sx1].copy()
    bg[dy0:dy1, dx0:dx1][mask] = src_patch[mask]


def heal_circle(
    bg: np.ndarray, *,
    src_x: int, src_y: int, dst_x: int, dst_y: int, radius: int,
) -> None:
    """Poisson-blend a circular source patch into ``bg`` at the destination.

    Wraps ``cv2.seamlessClone(NORMAL_CLONE)``. Silently skips when the
    source patch or the destination centre falls too close to the image
    edge for ``cv2.seamlessClone`` to handle (it requires a small margin
    on both sides).
    """
    if radius < 1:
        raise ValueError("radius must be >= 1.")
    if bg.dtype != np.uint8:
        raise ValueError("bg must be uint8.")
    h, w = bg.shape

    sx0, sy0 = src_x - radius, src_y - radius
    sx1, sy1 = src_x + radius + 1, src_y + radius + 1
    if sx0 < 0 or sy0 < 0 or sx1 > w or sy1 > h:
        return
    if (
        dst_x - radius - 1 < 0 or dst_x + radius + 1 >= w
        or dst_y - radius - 1 < 0 or dst_y + radius + 1 >= h
    ):
        return

    src_patch = bg[sy0:sy1, sx0:sx1]
    diameter = src_patch.shape[0]
    cmask = np.zeros((diameter, diameter), dtype=np.uint8)
    cv2.circle(cmask, (radius, radius), max(1, radius - 1), 255, thickness=-1)

    src_3ch = cv2.cvtColor(src_patch, cv2.COLOR_GRAY2BGR)
    dst_3ch = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    try:
        result = cv2.seamlessClone(
            src_3ch, dst_3ch, cmask, (dst_x, dst_y), cv2.NORMAL_CLONE,
        )
    except cv2.error:
        return
    np.copyto(bg, cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))


# ── Canvas ──────────────────────────────────────────────────────────────


class BgEditCanvas(QtWidgets.QLabel):
    """Custom ``QLabel`` that paints / stamps onto a working copy of a bg."""

    strokeFinished = QtCore.pyqtSignal(object)
    """Emitted with the pre-stroke bg snapshot once a clone / heal stroke ends."""

    def __init__(self, bg: np.ndarray, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._bg = bg.copy()
        self._mask = np.zeros(bg.shape, dtype=np.uint8)
        self._brush_size = DEFAULT_BRUSH_SIZE
        self._tool: ToolName = "smart_fill"
        self._mode = "idle"  # idle | paint_mask | erase_mask | stroke
        self._source_anchor: tuple[int, int] | None = None
        self._stroke_offset: tuple[int, int] | None = None
        self._stroke_pre_bg: np.ndarray | None = None
        self._cursor_pos: tuple[int, int] | None = None
        self.setMouseTracking(True)
        self.setFixedSize(bg.shape[1], bg.shape[0])
        self._refresh()

    # ── Public API ─────────────────────────────────────────────────────

    def set_tool(self, tool: ToolName) -> None:
        if tool not in ("smart_fill", "clone_stamp", "healing_brush"):
            raise ValueError(f"Unknown tool: {tool!r}.")
        self._tool = tool
        self._mode = "idle"
        self._refresh()

    def set_brush_size(self, size: int) -> None:
        self._brush_size = max(1, int(size))
        self._refresh()

    def current_bg(self) -> np.ndarray:
        return self._bg.copy()

    def current_mask(self) -> np.ndarray:
        return self._mask.copy()

    def has_mask(self) -> bool:
        return bool(self._mask.any())

    def clear_mask(self) -> None:
        self._mask[:] = 0
        self._refresh()

    def replace_bg(self, new_bg: np.ndarray) -> None:
        if new_bg.shape != self._bg.shape:
            raise ValueError(
                f"new bg shape {new_bg.shape} differs from current {self._bg.shape}.",
            )
        self._bg = new_bg.copy()
        self._mask[:] = 0
        self._refresh()

    # ── Mouse events ───────────────────────────────────────────────────

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        pos = self._image_pos(event)
        button = event.button()
        modifiers = event.modifiers()
        if button == Qt.MouseButton.LeftButton:
            if (
                modifiers & Qt.KeyboardModifier.AltModifier
                and self._tool in ("clone_stamp", "healing_brush")
            ):
                self._source_anchor = pos
                self._refresh()
                return
            if self._tool == "smart_fill":
                self._mode = "paint_mask"
                self._stamp_mask(pos, value=255)
            elif self._source_anchor is not None:
                self._begin_stroke(pos)
        elif button == Qt.MouseButton.RightButton and self._tool == "smart_fill":
            self._mode = "erase_mask"
            self._stamp_mask(pos, value=0)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        pos = self._image_pos(event)
        self._cursor_pos = pos
        if self._mode == "paint_mask":
            self._stamp_mask(pos, value=255)
        elif self._mode == "erase_mask":
            self._stamp_mask(pos, value=0)
        elif self._mode == "stroke":
            self._stamp_stroke(pos)
        else:
            self._refresh()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._mode == "stroke":
            self._end_stroke()
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._mode = "idle"

    def leaveEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802
        self._cursor_pos = None
        self._refresh()
        super().leaveEvent(event)

    # ── Internal ───────────────────────────────────────────────────────

    @staticmethod
    def _image_pos(event: QtGui.QMouseEvent) -> tuple[int, int]:
        return (int(event.position().x()), int(event.position().y()))

    def _stamp_mask(self, pos: tuple[int, int], *, value: int) -> None:
        cv2.circle(self._mask, pos, self._brush_size, value, thickness=-1)
        self._refresh()

    def _begin_stroke(self, pos: tuple[int, int]) -> None:
        if self._source_anchor is None:
            return
        ax, ay = self._source_anchor
        self._stroke_offset = (ax - pos[0], ay - pos[1])
        self._stroke_pre_bg = self._bg.copy()
        self._mode = "stroke"
        self._stamp_stroke(pos)

    def _end_stroke(self) -> None:
        if self._stroke_pre_bg is not None:
            self.strokeFinished.emit(self._stroke_pre_bg)
        self._stroke_pre_bg = None
        self._stroke_offset = None

    def _stamp_stroke(self, pos: tuple[int, int]) -> None:
        if self._stroke_offset is None:
            return
        ox, oy = self._stroke_offset
        sx, sy = pos[0] + ox, pos[1] + oy
        if self._tool == "clone_stamp":
            copy_circle(
                self._bg, src_x=sx, src_y=sy,
                dst_x=pos[0], dst_y=pos[1], radius=self._brush_size,
            )
        elif self._tool == "healing_brush":
            heal_circle(
                self._bg, src_x=sx, src_y=sy,
                dst_x=pos[0], dst_y=pos[1], radius=self._brush_size,
            )
        self._refresh()

    def _refresh(self) -> None:
        bgr = cv2.cvtColor(self._bg, cv2.COLOR_GRAY2BGR)
        if self._tool == "smart_fill" and self._mask.any():
            overlay = bgr.copy()
            overlay[self._mask > 0] = (0, 0, 220)
            bgr = cv2.addWeighted(bgr, 0.55, overlay, 0.45, 0)
        if (
            self._tool in ("clone_stamp", "healing_brush")
            and self._source_anchor is not None
        ):
            cv2.drawMarker(
                bgr, self._source_anchor, (50, 220, 50),
                cv2.MARKER_CROSS, 16, 2,
            )
        if self._cursor_pos is not None:
            cv2.circle(bgr, self._cursor_pos, self._brush_size, (255, 255, 255), 1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        image = QtGui.QImage(
            rgb.tobytes(), w, h, w * 3, QtGui.QImage.Format.Format_RGB888,
        )
        self.setPixmap(QtGui.QPixmap.fromImage(image))


# ── Dialog ──────────────────────────────────────────────────────────────


class BgEditDialog(QtWidgets.QDialog):
    """Modal dialog hosting the canvas plus a tool / action panel."""

    def __init__(
        self, initial_bg: np.ndarray, parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit background")
        self._undo: list[np.ndarray] = []
        self._redo: list[np.ndarray] = []

        self._canvas = BgEditCanvas(initial_bg, parent=self)
        self._canvas.strokeFinished.connect(self._on_stroke_finished)

        self._scroll = QtWidgets.QScrollArea(self)
        self._scroll.setWidget(self._canvas)
        self._scroll.setWidgetResizable(False)

        controls = self._build_controls()

        outer = QtWidgets.QHBoxLayout(self)
        outer.addWidget(self._scroll, stretch=1)
        outer.addWidget(controls)
        self.setLayout(outer)

        target_w = min(initial_bg.shape[1] + 360, 1500)
        target_h = min(initial_bg.shape[0] + 80, 950)
        self.resize(target_w, target_h)

    def result_bg(self) -> np.ndarray:
        return self._canvas.current_bg()

    # ── Layout ─────────────────────────────────────────────────────────

    def _build_controls(self) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget(self)
        v = QtWidgets.QVBoxLayout(wrap)
        v.setContentsMargins(8, 8, 8, 8)

        v.addWidget(self._build_tool_group(wrap))

        form = QtWidgets.QFormLayout()
        self._sb_brush = QtWidgets.QSpinBox(wrap)
        self._sb_brush.setRange(1, 200)
        self._sb_brush.setValue(DEFAULT_BRUSH_SIZE)
        self._sb_brush.valueChanged.connect(self._canvas.set_brush_size)
        form.addRow("Brush size (px)", self._sb_brush)

        self._sb_radius = QtWidgets.QSpinBox(wrap)
        self._sb_radius.setRange(1, 50)
        self._sb_radius.setValue(DEFAULT_INPAINT_RADIUS)
        form.addRow("Inpaint radius", self._sb_radius)
        v.addLayout(form)

        self._btn_inpaint = self._action_button(
            "Apply smart fill", self._on_inpaint,
        )
        self._btn_inpaint.setToolTip(
            "Fill the painted (red) region using surrounding pixels "
            "(cv2 TELEA inpaint). Use this AFTER painting a mask with the "
            "smart-fill brush.",
        )
        v.addWidget(self._btn_inpaint)
        v.addWidget(self._action_button("Clear mask", self._canvas.clear_mask))
        v.addWidget(
            self._action_button("Undo", self._on_undo, shortcut="Ctrl+Z"),
        )
        v.addWidget(
            self._action_button("Redo", self._on_redo, shortcut="Ctrl+Y"),
        )

        self._lbl_status = QtWidgets.QLabel("Ready.", wrap)
        self._lbl_status.setWordWrap(True)
        v.addWidget(self._lbl_status)

        v.addStretch(1)

        save = QtWidgets.QPushButton("Save && close", wrap)
        save.clicked.connect(self.accept)
        v.addWidget(save)
        cancel = QtWidgets.QPushButton("Cancel", wrap)
        cancel.clicked.connect(self.reject)
        v.addWidget(cancel)

        return wrap

    def _build_tool_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Tool", parent)
        v = QtWidgets.QVBoxLayout(box)
        self._rb_smart = QtWidgets.QRadioButton("Smart fill (paint + Inpaint)", box)
        self._rb_smart.setChecked(True)
        self._rb_clone = QtWidgets.QRadioButton("Clone stamp (Alt-click source)", box)
        self._rb_heal = QtWidgets.QRadioButton(
            "Healing brush (Alt-click source, blends)", box,
        )
        v.addWidget(self._rb_smart)
        v.addWidget(self._rb_clone)
        v.addWidget(self._rb_heal)

        self._lbl_hint = QtWidgets.QLabel(TOOL_HINTS["smart_fill"], box)
        self._lbl_hint.setWordWrap(True)
        self._lbl_hint.setStyleSheet("color: gray;")
        v.addWidget(self._lbl_hint)

        self._rb_smart.toggled.connect(
            lambda on: on and self._on_tool_changed("smart_fill"),
        )
        self._rb_clone.toggled.connect(
            lambda on: on and self._on_tool_changed("clone_stamp"),
        )
        self._rb_heal.toggled.connect(
            lambda on: on and self._on_tool_changed("healing_brush"),
        )
        return box

    def _action_button(
        self, label: str, slot, *, shortcut: str | None = None,
    ) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(label, self)
        if shortcut:
            btn.setShortcut(QtGui.QKeySequence(shortcut))
        btn.clicked.connect(slot)
        return btn

    # ── Slots ─────────────────────────────────────────────────────────

    def _on_tool_changed(self, tool: ToolName) -> None:
        self._canvas.set_tool(tool)
        self._lbl_hint.setText(TOOL_HINTS[tool])

    def _on_stroke_finished(self, prev_bg: np.ndarray) -> None:
        self._push_undo(prev_bg)
        self._redo.clear()
        self._set_status(
            f"Stroke applied. Undo: {len(self._undo)}.", kind="ok",
        )

    def _on_inpaint(self) -> None:
        if not self._canvas.has_mask():
            self._set_status(
                "Paint a mask first (LMB with the smart-fill tool).",
                kind="warn",
            )
            return
        bg = self._canvas.current_bg()
        mask = self._canvas.current_mask()
        masked_pixels = int((mask > 0).sum())
        self._push_undo(bg)
        self._redo.clear()
        try:
            new_bg = apply_inpaint(bg, mask, radius=self._sb_radius.value())
        except ValueError as exc:
            self._set_status(f"Inpaint failed: {exc}", kind="error")
            return
        self._canvas.replace_bg(new_bg)
        self._set_status(
            f"✓ Smart fill applied to {masked_pixels} px. "
            f"Undo stack: {len(self._undo)}.",
            kind="ok",
        )

    def _set_status(self, text: str, *, kind: str = "info") -> None:
        """Update the status label with a colour cue per ``kind``."""
        colour = {
            "ok":    "#1e8449",
            "warn":  "#b9770e",
            "error": "#c0392b",
            "info":  "#34495e",
        }.get(kind, "#34495e")
        self._lbl_status.setText(text)
        self._lbl_status.setStyleSheet(
            f"QLabel {{ color: {colour}; font-weight: bold; }}",
        )

    def _on_undo(self) -> None:
        if not self._undo:
            self._set_status("Nothing to undo.", kind="warn")
            return
        prev = self._undo.pop()
        self._redo.append(self._canvas.current_bg())
        self._canvas.replace_bg(prev)
        self._set_status(
            f"Undid one step. Undo: {len(self._undo)}, redo: {len(self._redo)}.",
            kind="ok",
        )

    def _on_redo(self) -> None:
        if not self._redo:
            self._set_status("Nothing to redo.", kind="warn")
            return
        nxt = self._redo.pop()
        self._push_undo(self._canvas.current_bg())
        self._canvas.replace_bg(nxt)
        self._set_status(
            f"Redid one step. Undo: {len(self._undo)}, redo: {len(self._redo)}.",
            kind="ok",
        )

    def _push_undo(self, bg: np.ndarray) -> None:
        self._undo.append(bg.copy())
        if len(self._undo) > HISTORY_LIMIT:
            self._undo.pop(0)


__all__ = [
    "BgEditCanvas",
    "BgEditDialog",
    "apply_inpaint",
    "copy_circle",
    "heal_circle",
]

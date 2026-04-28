"""Background editor: paint a mask, smart-fill the masked region.

Photoshop's "content-aware fill" via :func:`cv2.inpaint`. Useful for
removing a stationary fly (dead, decapitated, or one that simply does
not move enough during the recording) that the max-projection
background model has baked into itself.

The pure :func:`apply_inpaint` helper is testable without Qt; the
:class:`BgEditDialog` is the interactive front-end.
"""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

DEFAULT_BRUSH_SIZE = 12
DEFAULT_INPAINT_RADIUS = 3
HISTORY_LIMIT = 32


def apply_inpaint(
    bg: np.ndarray, mask: np.ndarray, radius: int = DEFAULT_INPAINT_RADIUS,
) -> np.ndarray:
    """Fill the masked region of ``bg`` using surrounding pixels.

    Args:
        bg: Grayscale ``uint8`` background.
        mask: ``uint8`` mask the same shape as ``bg``; non-zero pixels are
            filled.
        radius: Inpaint radius in pixels (TELEA algorithm).

    Returns:
        New ``uint8`` background with the masked region filled.

    Raises:
        ValueError: If shapes mismatch or dtypes are wrong.
    """
    if bg.shape != mask.shape:
        raise ValueError(f"Shape mismatch: bg {bg.shape} vs mask {mask.shape}.")
    if bg.dtype != np.uint8:
        raise ValueError("bg must be uint8.")
    if mask.dtype != np.uint8:
        raise ValueError("mask must be uint8.")
    if radius < 1:
        raise ValueError("radius must be >= 1.")
    return cv2.inpaint(bg, mask, radius, cv2.INPAINT_TELEA)


class BgEditCanvas(QtWidgets.QLabel):
    """Custom ``QLabel`` that paints a mask over a background image."""

    def __init__(self, bg: np.ndarray, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._bg = bg.copy()
        self._mask = np.zeros(bg.shape, dtype=np.uint8)
        self._brush_size = DEFAULT_BRUSH_SIZE
        self._painting = False
        self._erasing = False
        self.setMouseTracking(True)
        self.setFixedSize(bg.shape[1], bg.shape[0])
        self._refresh()

    # ── Public API ─────────────────────────────────────────────────────

    def set_brush_size(self, size: int) -> None:
        self._brush_size = max(1, int(size))

    def current_bg(self) -> np.ndarray:
        return self._bg.copy()

    def current_mask(self) -> np.ndarray:
        return self._mask.copy()

    def clear_mask(self) -> None:
        self._mask[:] = 0
        self._refresh()

    def replace_bg(self, new_bg: np.ndarray) -> None:
        """Replace the working background and reset the mask."""
        if new_bg.shape != self._bg.shape:
            raise ValueError(
                f"new bg shape {new_bg.shape} differs from current {self._bg.shape}.",
            )
        self._bg = new_bg.copy()
        self._mask[:] = 0
        self._refresh()

    def has_mask(self) -> bool:
        return bool(self._mask.any())

    # ── Mouse events ───────────────────────────────────────────────────

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._painting = True
            self._stamp(event.position(), value=255)
        elif event.button() == Qt.MouseButton.RightButton:
            self._erasing = True
            self._stamp(event.position(), value=0)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._painting:
            self._stamp(event.position(), value=255)
        elif self._erasing:
            self._stamp(event.position(), value=0)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._painting = False
        elif event.button() == Qt.MouseButton.RightButton:
            self._erasing = False

    # ── Internal ───────────────────────────────────────────────────────

    def _stamp(self, pos: QtCore.QPointF, *, value: int) -> None:
        x, y = int(pos.x()), int(pos.y())
        cv2.circle(self._mask, (x, y), self._brush_size, value, thickness=-1)
        self._refresh()

    def _refresh(self) -> None:
        bgr = cv2.cvtColor(self._bg, cv2.COLOR_GRAY2BGR)
        if self._mask.any():
            overlay = bgr.copy()
            overlay[self._mask > 0] = (0, 0, 220)  # red in BGR
            bgr = cv2.addWeighted(bgr, 0.55, overlay, 0.45, 0)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        image = QtGui.QImage(
            rgb.tobytes(), w, h, w * 3, QtGui.QImage.Format.Format_RGB888,
        )
        self.setPixmap(QtGui.QPixmap.fromImage(image))


class BgEditDialog(QtWidgets.QDialog):
    """Modal dialog for editing a background via paint + smart fill."""

    def __init__(
        self, initial_bg: np.ndarray, parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit background")
        self._undo: list[np.ndarray] = []
        self._redo: list[np.ndarray] = []

        self._canvas = BgEditCanvas(initial_bg, parent=self)
        self._scroll = QtWidgets.QScrollArea(self)
        self._scroll.setWidget(self._canvas)
        self._scroll.setWidgetResizable(False)

        controls = self._build_controls()

        outer = QtWidgets.QHBoxLayout(self)
        outer.addWidget(self._scroll, stretch=1)
        outer.addWidget(controls)
        self.setLayout(outer)

        target_w = min(initial_bg.shape[1] + 320, 1400)
        target_h = min(initial_bg.shape[0] + 80, 900)
        self.resize(target_w, target_h)

    # ── Public API ─────────────────────────────────────────────────────

    def result_bg(self) -> np.ndarray:
        """Return the edited background (call after :meth:`exec`)."""
        return self._canvas.current_bg()

    # ── Layout ─────────────────────────────────────────────────────────

    def _build_controls(self) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget(self)
        v = QtWidgets.QVBoxLayout(wrap)
        v.setContentsMargins(8, 8, 8, 8)

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

        v.addWidget(self._action_button("Inpaint mask", self._on_inpaint))
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

    def _action_button(
        self, label: str, slot, *, shortcut: str | None = None,
    ) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(label, self)
        if shortcut:
            btn.setShortcut(QtGui.QKeySequence(shortcut))
        btn.clicked.connect(slot)
        return btn

    # ── Actions ────────────────────────────────────────────────────────

    def _on_inpaint(self) -> None:
        if not self._canvas.has_mask():
            self._lbl_status.setText("Paint a mask first.")
            return
        bg = self._canvas.current_bg()
        mask = self._canvas.current_mask()
        self._push_undo(bg)
        self._redo.clear()
        try:
            new_bg = apply_inpaint(bg, mask, radius=self._sb_radius.value())
        except ValueError as exc:
            self._lbl_status.setText(f"Inpaint failed: {exc}")
            return
        self._canvas.replace_bg(new_bg)
        self._lbl_status.setText(
            f"Filled. Undo stack: {len(self._undo)}.",
        )

    def _on_undo(self) -> None:
        if not self._undo:
            self._lbl_status.setText("Nothing to undo.")
            return
        prev = self._undo.pop()
        self._redo.append(self._canvas.current_bg())
        self._canvas.replace_bg(prev)
        self._lbl_status.setText(
            f"Undid one step. Undo: {len(self._undo)}, redo: {len(self._redo)}.",
        )

    def _on_redo(self) -> None:
        if not self._redo:
            self._lbl_status.setText("Nothing to redo.")
            return
        nxt = self._redo.pop()
        self._push_undo(self._canvas.current_bg())
        self._canvas.replace_bg(nxt)
        self._lbl_status.setText(
            f"Redid one step. Undo: {len(self._undo)}, redo: {len(self._redo)}.",
        )

    def _push_undo(self, bg: np.ndarray) -> None:
        self._undo.append(bg.copy())
        if len(self._undo) > HISTORY_LIMIT:
            self._undo.pop(0)


__all__ = ["BgEditCanvas", "BgEditDialog", "apply_inpaint"]

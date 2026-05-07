"""FrameCanvas circle-handle drag behaviour — pin the bug fix."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6 import QtWidgets

from pylace.annotator.geometry import Circle
from pylace.annotator.main_window import FrameCanvas


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _build_canvas_with_circle(app, cx=200.0, cy=200.0, r=80.0) -> FrameCanvas:
    canvas = FrameCanvas()
    canvas.set_frame(np.zeros((400, 400, 3), dtype=np.uint8))
    canvas.set_arena(Circle(cx=cx, cy=cy, r=r))
    canvas.set_mode("edit")
    return canvas


def test_grabbing_centre_handle_drag_kind_is_centre(app):
    """Click on the centre anchor → drag_kind must be 'centre'.

    Regression: a stale CIRCLE_HANDLE_LABELS tuple was indexed in an
    order incompatible with Circle.origin_candidates(), so clicking
    the centre got drag_kind='south' and immediately resized the
    circle to 1 px instead of moving it.
    """
    canvas = _build_canvas_with_circle(app)
    canvas._maybe_grab_handle((200.0, 200.0))  # centre
    assert canvas._drag_kind == "centre"


def test_grabbing_east_handle_drag_kind_is_east(app):
    """Click on the east rim → drag_kind must be 'east' (not 'centre')."""
    canvas = _build_canvas_with_circle(app)
    canvas._maybe_grab_handle((280.0, 200.0))  # east rim
    assert canvas._drag_kind == "east"


def test_grabbing_centre_then_dragging_translates_the_circle(app):
    """Grab centre, drag to (300, 250) → circle re-centres there at same r."""
    canvas = _build_canvas_with_circle(app, cx=200.0, cy=200.0, r=80.0)
    canvas._maybe_grab_handle((200.0, 200.0))
    canvas._update_dragged_handle((300.0, 250.0))
    arena = canvas.arena()
    assert isinstance(arena, Circle)
    assert arena.cx == pytest.approx(300.0)
    assert arena.cy == pytest.approx(250.0)
    assert arena.r == pytest.approx(80.0)


def test_grabbing_east_then_dragging_resizes_the_circle(app):
    """Grab east, drag outward → radius grows but centre stays put."""
    canvas = _build_canvas_with_circle(app, cx=200.0, cy=200.0, r=80.0)
    canvas._maybe_grab_handle((280.0, 200.0))  # east rim, r=80 from centre
    canvas._update_dragged_handle((350.0, 200.0))  # 150 px from centre
    arena = canvas.arena()
    assert isinstance(arena, Circle)
    assert arena.cx == pytest.approx(200.0)
    assert arena.cy == pytest.approx(200.0)
    assert arena.r == pytest.approx(150.0)

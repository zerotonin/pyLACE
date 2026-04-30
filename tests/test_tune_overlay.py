"""Overlay rendering on a synthetic frame."""

from __future__ import annotations

import cv2
import numpy as np

from pylace.annotator.geometry import Circle, Polygon
from pylace.detect.frame import Detection
from pylace.tune.overlay import (
    ARENA_COLOR,
    CENTROID_COLOR,
    ELLIPSE_COLOR,
    render_overlay,
)


def _gray_canvas(h: int = 80, w: int = 80, level: int = 200) -> np.ndarray:
    return np.full((h, w), level, dtype=np.uint8)


def _detection(cx: float = 40, cy: float = 40) -> Detection:
    return Detection(
        cx=cx, cy=cy,
        area_px=80.0, perimeter_px=35.0,
        major_axis_px=12.0, minor_axis_px=5.0,
        orientation_deg=0.0, contour=None,
    )


def test_overlay_returns_bgr_image_of_same_size():
    frame = _gray_canvas()
    arena = Circle(cx=40, cy=40, r=20)
    out = render_overlay(frame, arena, [])
    assert out.shape == (80, 80, 3)
    assert out.dtype == np.uint8


def test_overlay_marks_centroid_pixel_red():
    frame = _gray_canvas()
    arena = Circle(cx=40, cy=40, r=30)
    out = render_overlay(frame, arena, [_detection(cx=40, cy=40)])
    # Red marker hits the centroid pixel.
    assert tuple(out[40, 40]) == CENTROID_COLOR


def test_overlay_draws_arena_outline():
    frame = _gray_canvas()
    arena = Circle(cx=40, cy=40, r=20)
    out = render_overlay(frame, arena, [], show_centroids=False, show_ellipses=False)
    rim_pixels = (
        (out == np.array(ARENA_COLOR)).all(axis=2)
    ).sum()
    assert rim_pixels > 30  # some non-trivial number of cyan pixels


def test_overlay_ellipse_is_drawn_when_no_contour():
    frame = _gray_canvas()
    arena = Circle(cx=40, cy=40, r=30)
    out = render_overlay(frame, arena, [_detection()], show_arena=False, show_centroids=False)
    ellipse_hits = ((out == np.array(ELLIPSE_COLOR)).all(axis=2)).sum()
    assert ellipse_hits > 0


def test_overlay_foreground_mask_tints_pixels():
    frame = _gray_canvas(level=100)
    arena = Circle(cx=40, cy=40, r=30)
    mask = np.zeros((80, 80), dtype=bool)
    mask[10:20, 10:20] = True
    out = render_overlay(
        frame, arena, [],
        foreground_mask=mask,
        show_arena=False, show_centroids=False, show_ellipses=False,
    )
    # Tinted region differs from the untinted region.
    assert not np.array_equal(out[15, 15], out[60, 60])


def test_overlay_polygon_arena():
    frame = _gray_canvas()
    arena = Polygon([(10.0, 10.0), (70.0, 10.0), (70.0, 70.0), (10.0, 70.0)])
    out = render_overlay(frame, arena, [], show_centroids=False, show_ellipses=False)
    cyan_hits = ((out == np.array(ARENA_COLOR)).all(axis=2)).sum()
    assert cyan_hits > 100


def test_overlay_with_contour_draws_green_line():
    frame = _gray_canvas()
    arena = Circle(40, 40, 30)
    contour = np.array([[35, 38], [45, 38], [45, 42], [35, 42]], dtype=np.int32)
    contour = contour.reshape(-1, 1, 2)
    det = Detection(
        cx=40, cy=40, area_px=40, perimeter_px=20,
        major_axis_px=10, minor_axis_px=4,
        orientation_deg=0,
        contour=contour,
    )
    out = render_overlay(
        frame, arena, [det],
        show_arena=False, show_centroids=False, show_ellipses=False,
    )
    # Green contour pixels exist on the rectangle outline.
    green_hits = ((out == np.array((0, 255, 0))).all(axis=2)).sum()
    assert green_hits > 0

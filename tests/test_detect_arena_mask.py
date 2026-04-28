"""Arena-mask correctness for circle / rectangle / polygon."""

from __future__ import annotations

import numpy as np

from pylace.annotator.geometry import Circle, Polygon, Rectangle
from pylace.detect.arena_mask import arena_mask


def test_circle_mask_has_correct_area():
    circle = Circle(cx=50.0, cy=50.0, r=20.0)
    mask = arena_mask(circle, frame_size=(100, 100))
    expected_area = np.pi * 20.0 ** 2
    actual_area = float(mask.sum())
    # Discrete sampling tolerance: ±3 % of the analytical area.
    assert actual_area == ((100 * 100) - (mask == 0).sum())  # tautology sanity
    assert abs(actual_area - expected_area) / expected_area < 0.03


def test_circle_mask_centre_is_inside_and_far_corner_is_outside():
    circle = Circle(cx=50.0, cy=50.0, r=10.0)
    mask = arena_mask(circle, frame_size=(100, 100))
    assert mask[50, 50]
    assert not mask[0, 0]
    assert not mask[99, 99]


def test_rectangle_mask_is_full_axis_aligned_box():
    rect = Rectangle.from_two_points((10.0, 20.0), (60.0, 80.0))
    mask = arena_mask(rect, frame_size=(100, 100))
    inside = mask[20:81, 10:61]
    assert inside.all()
    assert not mask[0, 0]
    assert not mask[20, 5]


def test_polygon_mask_triangle():
    triangle = Polygon([(10.0, 10.0), (90.0, 10.0), (50.0, 90.0)])
    mask = arena_mask(triangle, frame_size=(100, 100))
    # Triangle with base 80 along the top, height 80: area ≈ 3200.
    assert 2800 < mask.sum() < 3500
    # Apex pixel inside.
    assert mask[50, 50]
    # Outside the triangle.
    assert not mask[80, 90]


def test_mask_shape_matches_frame_size():
    circle = Circle(0, 0, 1)
    mask = arena_mask(circle, frame_size=(640, 480))
    assert mask.shape == (480, 640)
    assert mask.dtype == bool

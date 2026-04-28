"""Geometry, world-frame, and calibration unit tests."""

from __future__ import annotations

import math

import pytest

from pylace.annotator.geometry import (
    Calibration,
    Circle,
    Polygon,
    Rectangle,
    WorldFrame,
    edge_length,
    pixel_to_world,
    shape_name,
)


def test_circle_origin_candidates_are_at_cardinals_and_centre():
    c = Circle(cx=10.0, cy=20.0, r=5.0)
    cands = dict(c.origin_candidates())
    assert cands["east"] == (15.0, 20.0)
    assert cands["north"] == (10.0, 15.0)
    assert cands["west"] == (5.0, 20.0)
    assert cands["south"] == (10.0, 25.0)
    assert cands["centre"] == (10.0, 20.0)


def test_rectangle_from_two_points_sorts_corners_axis_aligned():
    rect = Rectangle.from_two_points((10.0, 30.0), (5.0, 10.0))
    assert rect.vertices == [(5.0, 10.0), (10.0, 10.0), (10.0, 30.0), (5.0, 30.0)]


def test_rectangle_rejects_wrong_vertex_count():
    with pytest.raises(ValueError):
        Rectangle([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])


def test_polygon_rejects_too_few_vertices():
    with pytest.raises(ValueError):
        Polygon([(0.0, 0.0), (1.0, 1.0)])


def test_polygon_origin_candidates_label_each_vertex():
    poly = Polygon([(0.0, 0.0), (3.0, 0.0), (3.0, 4.0), (0.0, 4.0)])
    labels = [label for label, _ in poly.origin_candidates()]
    assert labels == ["vertex_0", "vertex_1", "vertex_2", "vertex_3"]


def test_shape_name_dispatches_correctly():
    assert shape_name(Circle(0, 0, 1)) == "circle"
    assert shape_name(Rectangle.from_two_points((0, 0), (1, 1))) == "rectangle"
    assert shape_name(Polygon([(0, 0), (1, 0), (0, 1)])) == "polygon"


def test_edge_length_is_pythagorean():
    verts = [(0.0, 0.0), (3.0, 4.0), (3.0, 0.0)]
    assert edge_length(verts, 0, 1) == pytest.approx(5.0)
    assert edge_length(verts, 0, 2) == pytest.approx(3.0)


def test_calibration_mm_per_pixel_is_ratio():
    cal = Calibration(reference_kind="diameter", physical_mm=30.0, pixel_distance=600.0)
    assert cal.mm_per_pixel == pytest.approx(0.05)


def test_calibration_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        Calibration(reference_kind="diameter", physical_mm=0.0, pixel_distance=10.0)
    with pytest.raises(ValueError):
        Calibration(reference_kind="diameter", physical_mm=10.0, pixel_distance=0.0)


def test_calibration_edge_requires_vertex_pair():
    with pytest.raises(ValueError):
        Calibration(reference_kind="edge", physical_mm=10.0, pixel_distance=100.0)


def test_calibration_diameter_must_not_set_vertex_pair():
    with pytest.raises(ValueError):
        Calibration(
            reference_kind="diameter",
            physical_mm=10.0,
            pixel_distance=100.0,
            reference_vertices=(0, 1),
        )


def test_pixel_to_world_y_up_x_right():
    frame = WorldFrame(origin_pixel=(100.0, 200.0), y_axis="up", x_axis="right")
    cal = Calibration(reference_kind="diameter", physical_mm=10.0, pixel_distance=100.0)
    # 50 px right, 50 px below origin in image coords = (50, 50) in image space.
    # mm_per_pixel = 0.1; in y-up world, "below origin" means negative y.
    result = pixel_to_world((150.0, 250.0), frame, cal)
    assert result == pytest.approx((5.0, -5.0))


def test_pixel_to_world_y_down_x_left():
    frame = WorldFrame(origin_pixel=(100.0, 200.0), y_axis="down", x_axis="left")
    cal = Calibration(reference_kind="diameter", physical_mm=10.0, pixel_distance=100.0)
    result = pixel_to_world((150.0, 250.0), frame, cal)
    # x_left flips x; y_down means image y increases ⇒ world y increases.
    assert result == pytest.approx((-5.0, 5.0))


def test_pixel_to_world_origin_maps_to_zero():
    frame = WorldFrame(origin_pixel=(123.4, 567.8), y_axis="up", x_axis="right")
    cal = Calibration(reference_kind="diameter", physical_mm=1.0, pixel_distance=1.0)
    assert pixel_to_world((123.4, 567.8), frame, cal) == pytest.approx((0.0, 0.0))


def test_pixel_to_world_uses_distance_via_mm_per_pixel():
    frame = WorldFrame(origin_pixel=(0.0, 0.0), y_axis="down", x_axis="right")
    cal = Calibration(reference_kind="edge", physical_mm=20.0, pixel_distance=100.0,
                      reference_vertices=(0, 1))
    x, y = pixel_to_world((10.0, 20.0), frame, cal)
    assert math.hypot(x, y) == pytest.approx(math.hypot(10.0, 20.0) * 0.2)

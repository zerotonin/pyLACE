"""Distance-to-wall + other derived columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pylace.annotator.geometry import Circle, Polygon, Rectangle
from pylace.posthoc.analytics import compute_distance_to_wall


def _track_df(cx: np.ndarray, cy: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "frame_idx": np.arange(cx.size),
        "track_id": [0] * cx.size,
        "cx_smooth_px": cx,
        "cy_smooth_px": cy,
    })


def test_distance_to_wall_circle_at_centre_equals_radius():
    arena = Circle(cx=100.0, cy=100.0, r=50.0)
    df = _track_df(np.full(5, 100.0), np.full(5, 100.0))
    d = compute_distance_to_wall(df, arena, pix_per_mm=10.0)
    assert np.allclose(d, 5.0)  # 50 px / 10 = 5 mm


def test_distance_to_wall_circle_on_boundary_is_zero():
    arena = Circle(cx=100.0, cy=100.0, r=50.0)
    df = _track_df(np.full(3, 150.0), np.full(3, 100.0))  # exactly r away
    d = compute_distance_to_wall(df, arena, pix_per_mm=10.0)
    assert np.allclose(d, 0.0, atol=1e-9)


def test_distance_to_wall_circle_outside_is_negative():
    arena = Circle(cx=100.0, cy=100.0, r=50.0)
    df = _track_df(np.full(3, 160.0), np.full(3, 100.0))  # 10 px outside
    d = compute_distance_to_wall(df, arena, pix_per_mm=10.0)
    assert np.allclose(d, -1.0)


def test_distance_to_wall_rectangle():
    arena = Rectangle.from_two_points((0.0, 0.0), (200.0, 100.0))
    # Centre at (100, 50). Nearest edge from centre is 50 px (top/bottom).
    df = _track_df(np.array([100.0, 100.0, 10.0]),
                   np.array([50.0,  90.0, 50.0]))
    d = compute_distance_to_wall(df, arena, pix_per_mm=10.0)
    # Centroid 50 px from edge → 5 mm. Near top: 10 px → 1 mm. Near
    # left edge: 10 px → 1 mm.
    assert d[0] == 5.0
    assert d[1] == 1.0
    assert d[2] == 1.0


def test_distance_to_wall_polygon():
    # Square, side 100, corners at (0,0)-(100,100).
    arena = Polygon(vertices=[(0.0, 0.0), (100.0, 0.0),
                              (100.0, 100.0), (0.0, 100.0)])
    df = _track_df(np.array([50.0, 5.0]), np.array([50.0, 50.0]))
    d = compute_distance_to_wall(df, arena, pix_per_mm=1.0)
    assert d[0] == 50.0   # centre, distance 50 px
    assert d[1] == 5.0    # 5 px from left edge


def test_distance_to_wall_handles_nan_centroids():
    arena = Circle(cx=100.0, cy=100.0, r=50.0)
    df = _track_df(np.array([100.0, np.nan, 100.0]),
                   np.array([100.0, np.nan, 100.0]))
    d = compute_distance_to_wall(df, arena, pix_per_mm=10.0)
    assert np.isnan(d[1])
    assert d[0] == 5.0
    assert d[2] == 5.0

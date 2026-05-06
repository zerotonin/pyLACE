"""Phase 2 — kinematic readouts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pylace.annotator.geometry import Circle, Polygon, Rectangle
from pylace.posthoc.metrics import (
    occupancy_heatmap,
    speed_summary,
    summarise_track,
    summarise_tracks,
    thigmotaxis_fraction,
    walk_stop_bouts,
    yaw_rate_summary,
)


def _track_df(
    *,
    track_id: int = 0,
    speed: np.ndarray | None = None,
    cx: np.ndarray | None = None,
    cy: np.ndarray | None = None,
    yaw_rate: np.ndarray | None = None,
) -> pd.DataFrame:
    n = (speed.size if speed is not None
         else cx.size if cx is not None
         else yaw_rate.size if yaw_rate is not None
         else 0)
    if n == 0:
        return pd.DataFrame(columns=[
            "frame_idx", "track_id", "cx_smooth_px", "cy_smooth_px",
            "speed_mm_s", "yaw_rate_deg_s",
        ])
    return pd.DataFrame({
        "frame_idx": np.arange(n),
        "track_id": [track_id] * n,
        "cx_smooth_px": cx if cx is not None else np.full(n, np.nan),
        "cy_smooth_px": cy if cy is not None else np.full(n, np.nan),
        "speed_mm_s": speed if speed is not None else np.full(n, np.nan),
        "yaw_rate_deg_s": yaw_rate if yaw_rate is not None else np.full(n, np.nan),
    })


# ─────────────────────────────────────────────────────────────────
#  Speed summary
# ─────────────────────────────────────────────────────────────────


def test_speed_summary_recovers_known_distribution():
    speed = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
    df = _track_df(speed=speed)
    s = speed_summary(df, fps=10.0)
    assert s["speed_median_mm_s"] == pytest.approx(3.5)
    assert s["speed_p90_mm_s"] == pytest.approx(7.5, abs=0.5)  # numpy p90 interpolates
    assert s["speed_max_mm_s"] == pytest.approx(10.0)
    # path = sum / fps = 25 / 10 = 2.5 mm
    assert s["path_length_mm"] == pytest.approx(2.5)


def test_speed_summary_all_nan_returns_nan():
    df = _track_df(speed=np.full(10, np.nan))
    s = speed_summary(df, fps=10.0)
    assert all(np.isnan(v) for v in s.values())


# ─────────────────────────────────────────────────────────────────
#  Walk / stop bouts
# ─────────────────────────────────────────────────────────────────


def test_walk_stop_bouts_finds_a_single_walking_bout():
    # 1 s standing, 2 s walking at 5 mm/s, 1 s standing.
    fps = 30.0
    speed = np.concatenate([
        np.full(int(fps), 0.0),
        np.full(int(2 * fps), 5.0),
        np.full(int(fps), 0.0),
    ])
    df = _track_df(speed=speed)
    bouts = walk_stop_bouts(df, fps=fps)
    assert len(bouts) == 1
    assert bouts.iloc[0]["duration_s"] == pytest.approx(2.0, abs=1.0 / fps)
    assert bouts.iloc[0]["mean_speed_mm_s"] == pytest.approx(5.0)


def test_walk_stop_bouts_hysteresis_does_not_fragment():
    """Speed jitters around the on-threshold but stays > off-threshold."""
    fps = 30.0
    base = 3.0  # above on
    jitter = np.array([1.5, 3.5, 1.8, 4.0, 1.2, 3.0] * 10)  # bounces but always > off
    speed = np.concatenate([np.full(15, 0.0), jitter, np.full(15, 0.0)])
    df = _track_df(speed=speed)
    bouts = walk_stop_bouts(
        df, fps=fps, on_threshold_mm_s=2.5, off_threshold_mm_s=1.0,
    )
    # Hysteresis should keep this as ONE bout (the jitter never drops below 1.0).
    assert len(bouts) == 1


def test_walk_stop_bouts_drops_short_walks():
    fps = 30.0
    # 2 frames of walking, then long rest. duration = 2/30 = 0.067 s, < 0.1 s.
    speed = np.zeros(60)
    speed[10:12] = 5.0
    df = _track_df(speed=speed)
    bouts = walk_stop_bouts(df, fps=fps, min_duration_s=0.1)
    assert len(bouts) == 0


def test_walk_stop_bouts_validates_thresholds():
    df = _track_df(speed=np.zeros(10))
    with pytest.raises(ValueError, match="must be >="):
        walk_stop_bouts(df, fps=10.0, on_threshold_mm_s=1.0, off_threshold_mm_s=2.0)


# ─────────────────────────────────────────────────────────────────
#  Occupancy
# ─────────────────────────────────────────────────────────────────


def test_occupancy_heatmap_concentrates_at_known_point():
    # All positions at (50, 100) in a 200 × 200 frame.
    n = 100
    cx = np.full(n, 50.0)
    cy = np.full(n, 100.0)
    df = _track_df(cx=cx, cy=cy)
    h = occupancy_heatmap(df, frame_size=(200, 200), n_bins=10)
    assert h.shape == (10, 10)
    assert h.sum() == pytest.approx(1.0)
    # Hottest bin should be the one containing the point.
    iy_hot, ix_hot = np.unravel_index(int(np.argmax(h)), h.shape)
    # (50, 100) → x bin 2, y bin 5 (out of 10 each, frame 200×200).
    assert ix_hot == 2
    assert iy_hot == 5


def test_occupancy_heatmap_empty_track_returns_zeros():
    df = _track_df(cx=np.full(5, np.nan), cy=np.full(5, np.nan))
    h = occupancy_heatmap(df, frame_size=(64, 64), n_bins=8)
    assert h.shape == (8, 8)
    assert h.sum() == 0.0


# ─────────────────────────────────────────────────────────────────
#  Thigmotaxis
# ─────────────────────────────────────────────────────────────────


def test_thigmotaxis_circle_perfectly_central():
    arena = Circle(cx=100.0, cy=100.0, r=50.0)
    df = _track_df(cx=np.full(100, 100.0), cy=np.full(100, 100.0))
    assert thigmotaxis_fraction(df, arena) == 0.0


def test_thigmotaxis_circle_perfectly_peripheral():
    arena = Circle(cx=100.0, cy=100.0, r=50.0)
    # All positions at distance 49 (just inside r); 49 > 0.8 * 50 = 40 → all peripheral.
    df = _track_df(cx=np.full(100, 149.0), cy=np.full(100, 100.0))
    assert thigmotaxis_fraction(df, arena, outer_band_frac=0.20) == 1.0


def test_thigmotaxis_circle_half_and_half():
    arena = Circle(cx=100.0, cy=100.0, r=50.0)
    cx = np.concatenate([np.full(50, 100.0), np.full(50, 145.0)])  # 50 central, 50 outer
    cy = np.full(100, 100.0)
    df = _track_df(cx=cx, cy=cy)
    assert thigmotaxis_fraction(df, arena, outer_band_frac=0.20) == pytest.approx(0.5)


def test_thigmotaxis_rectangle_uses_inscribed_radius():
    arena = Rectangle.from_two_points((0.0, 0.0), (200.0, 100.0))
    # Centre at (100, 50). Inscribed radius = min(100, 50) = 50.
    # outer band starts at d = 50 * 0.8 = 40.
    cx = np.array([100.0, 145.0])
    cy = np.array([50.0, 50.0])
    df = _track_df(cx=cx, cy=cy)
    # First point central (d=0); second point d=45 > 40 → peripheral.
    f = thigmotaxis_fraction(df, arena, outer_band_frac=0.20)
    assert f == pytest.approx(0.5)


def test_thigmotaxis_polygon_uses_inscribed_radius():
    # Hexagon inscribed in a 100-radius circle, centre at (200, 200).
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    verts = [(200 + 100 * np.cos(a), 200 + 100 * np.sin(a)) for a in angles]
    arena = Polygon(vertices=verts)
    df = _track_df(cx=np.full(10, 200.0), cy=np.full(10, 200.0))
    assert thigmotaxis_fraction(df, arena) == 0.0


# ─────────────────────────────────────────────────────────────────
#  Yaw rate
# ─────────────────────────────────────────────────────────────────


def test_yaw_rate_summary_recovers_known_distribution():
    yaw_rate = np.array([-50.0, -10.0, 0.0, 10.0, 50.0])
    df = _track_df(yaw_rate=yaw_rate)
    s = yaw_rate_summary(df)
    # |yaw_rate| = [50, 10, 0, 10, 50] → median 10, p95 ≈ 50
    assert s["yaw_rate_median_abs_deg_s"] == pytest.approx(10.0)
    assert s["yaw_rate_p95_abs_deg_s"] == pytest.approx(50.0, abs=1.0)


def test_yaw_rate_summary_all_nan_returns_nan():
    df = _track_df(yaw_rate=np.full(5, np.nan))
    s = yaw_rate_summary(df)
    assert all(np.isnan(v) for v in s.values())


# ─────────────────────────────────────────────────────────────────
#  Aggregator
# ─────────────────────────────────────────────────────────────────


def test_summarise_track_combines_all_readouts():
    fps = 30.0
    n = int(3 * fps)  # 3 seconds
    speed = np.full(n, 5.0)  # constantly walking
    yaw_rate = np.full(n, 0.0)
    cx = np.linspace(50, 200, n)
    cy = np.full(n, 100.0)
    df = _track_df(speed=speed, yaw_rate=yaw_rate, cx=cx, cy=cy)
    arena = Circle(cx=125.0, cy=100.0, r=80.0)
    summary = summarise_track(df, fps=fps, arena=arena)
    assert summary["track_id"] == 0
    assert summary["n_frames"] == n
    assert summary["total_time_s"] == pytest.approx(3.0)
    assert summary["speed_median_mm_s"] == pytest.approx(5.0)
    assert summary["n_walk_bouts"] == 1
    assert summary["walking_fraction"] == pytest.approx(1.0)
    assert summary["yaw_rate_median_abs_deg_s"] == pytest.approx(0.0)
    assert "thigmotaxis_fraction" in summary


def test_summarise_tracks_returns_one_row_per_track():
    fps = 30.0
    n = 60
    speed_a = np.full(n, 5.0)
    speed_b = np.full(n, 0.0)
    cx = np.full(n, 100.0)
    cy = np.full(n, 100.0)
    yaw_rate = np.full(n, 0.0)
    df_a = _track_df(track_id=0, speed=speed_a, cx=cx, cy=cy, yaw_rate=yaw_rate)
    df_b = _track_df(track_id=1, speed=speed_b, cx=cx, cy=cy, yaw_rate=yaw_rate)
    df = pd.concat([df_a, df_b], ignore_index=True)
    summary = summarise_tracks(df, fps=fps)
    assert len(summary) == 2
    assert sorted(summary["track_id"].tolist()) == [0, 1]
    by_id = summary.set_index("track_id")
    assert by_id.loc[0, "walking_fraction"] == pytest.approx(1.0)
    assert by_id.loc[1, "walking_fraction"] == pytest.approx(0.0)

"""Trajectory cleaner: gap fill, outlier rejection, Savitzky-Golay smoothing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pylace.posthoc.clean import (
    _interpolate_with_max_gap,
    _safe_gradient,
    _sg_smooth_runs,
    clean_trajectory,
)


def _synth_track_df(
    cx: np.ndarray, cy: np.ndarray, *,
    track_id: int = 0,
    frame_offset: int = 0,
    orientation_deg: float | np.ndarray = 0.0,
) -> pd.DataFrame:
    """Build a minimal detection DataFrame from arrays of cx, cy."""
    n = cx.size
    if np.isscalar(orientation_deg):
        orientation = np.full(n, float(orientation_deg))
    else:
        orientation = orientation_deg
    return pd.DataFrame({
        "frame_idx": np.arange(n) + frame_offset,
        "roi_label": ["_merged"] * n,
        "track_id": [track_id] * n,
        "cx_px": cx,
        "cy_px": cy,
        "x_mm": cx * 0.1,
        "y_mm": cy * 0.1,
        "area_px": np.full(n, 2500.0),
        "perimeter_px": np.full(n, 180.0),
        "solidity": np.full(n, 0.92),
        "major_axis_px": np.full(n, 58.0),
        "minor_axis_px": np.full(n, 22.0),
        "orientation_deg": orientation,
    })


# ─────────────────────────────────────────────────────────────────
#  Gap fill
# ─────────────────────────────────────────────────────────────────


def test_interpolate_fills_short_gaps():
    arr = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
    out = _interpolate_with_max_gap(arr, max_gap=3)
    assert np.allclose(out, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_interpolate_leaves_long_gaps_alone():
    arr = np.array([1.0, np.nan, np.nan, np.nan, np.nan, 6.0])
    out = _interpolate_with_max_gap(arr, max_gap=3)
    assert np.isnan(out[1:5]).all()
    assert out[0] == 1.0 and out[5] == 6.0


def test_interpolate_leaves_unbounded_runs_alone():
    arr = np.array([np.nan, np.nan, 3.0, 4.0])
    out = _interpolate_with_max_gap(arr, max_gap=10)
    assert np.isnan(out[:2]).all()
    assert out[2] == 3.0 and out[3] == 4.0


# ─────────────────────────────────────────────────────────────────
#  SG smoothing
# ─────────────────────────────────────────────────────────────────


def test_sg_smoothing_reduces_gaussian_noise():
    rng = np.random.default_rng(0)
    n = 200
    x = np.linspace(0, 10, n)
    clean = np.sin(x)
    noisy = clean + rng.normal(0, 0.3, n)
    smoothed = _sg_smooth_runs(noisy, window=11, polyorder=3)
    assert np.std(smoothed - clean) < np.std(noisy - clean) * 0.5


def test_sg_skips_short_runs():
    arr = np.array([1.0, 2.0, 3.0, np.nan, np.nan, 4.0, 5.0])
    # Window is too long for either run; both run untouched.
    out = _sg_smooth_runs(arr, window=11, polyorder=3)
    assert np.allclose(out[:3], arr[:3])
    assert np.allclose(out[5:], arr[5:])


def test_sg_window_validation():
    import pytest
    with pytest.raises(ValueError):
        _sg_smooth_runs(np.zeros(20), window=3, polyorder=3)
    with pytest.raises(ValueError):
        _sg_smooth_runs(np.zeros(20), window=10, polyorder=3)


# ─────────────────────────────────────────────────────────────────
#  Safe gradient
# ─────────────────────────────────────────────────────────────────


def test_safe_gradient_propagates_nan_runs():
    arr = np.array([0.0, 1.0, 2.0, np.nan, 5.0, 6.0])
    grad = _safe_gradient(arr, dt=1.0)
    assert np.allclose(grad[:3], [1.0, 1.0, 1.0])
    assert np.isnan(grad[3])
    assert np.allclose(grad[4:], [1.0, 1.0])


# ─────────────────────────────────────────────────────────────────
#  End-to-end clean_trajectory
# ─────────────────────────────────────────────────────────────────


def test_clean_track_fills_short_gap_and_marks_interpolated():
    n = 30
    cx = np.linspace(0, 290, n)
    cy = np.full(n, 100.0)
    df = _synth_track_df(cx, cy)
    # Drop frames 5–7 to create a 3-frame gap.
    df = df.drop(index=[5, 6, 7]).reset_index(drop=True)
    cleaned = clean_trajectory(df, fps=30.0, pix_per_mm=10.0,
                               max_gap_frames=5, sg_window=5, sg_polyorder=2)
    assert len(cleaned) == n
    assert cleaned.loc[5:7, "interpolated"].all()
    assert cleaned.loc[5:7, "cx_px"].notna().all()
    assert not cleaned.loc[8:, "interpolated"].any()


def test_clean_track_flags_outlier_when_velocity_exceeds_threshold():
    n = 20
    cx = np.linspace(0, 19, n)
    cy = np.full(n, 50.0)
    cx[10] = 5000.0  # huge teleport
    df = _synth_track_df(cx, cy)
    cleaned = clean_trajectory(
        df, fps=30.0, pix_per_mm=10.0,
        max_gap_frames=5, max_speed_mm_s=200.0,
        sg_window=5, sg_polyorder=2,
    )
    assert bool(cleaned.loc[10, "outlier_rejected"]) is True
    # The teleport should have been replaced with the interpolation.
    assert cleaned.loc[10, "cx_px"] < 100.0


def test_clean_track_smoothes_noisy_positions():
    rng = np.random.default_rng(42)
    n = 200
    t = np.arange(n)
    cx_clean = t * 1.0
    cy_clean = 10 * np.sin(t / 10.0)
    cx = cx_clean + rng.normal(0, 0.5, n)
    cy = cy_clean + rng.normal(0, 0.5, n)
    df = _synth_track_df(cx, cy)
    cleaned = clean_trajectory(
        df, fps=30.0, pix_per_mm=10.0,
        sg_window=11, sg_polyorder=3,
    )
    err_raw = np.std(cx - cx_clean)
    err_smoothed = np.std(cleaned["cx_smooth_px"].to_numpy() - cx_clean)
    assert err_smoothed < err_raw * 0.5


def test_clean_track_computes_velocity_in_mm_per_s():
    n = 60
    # 1 px / frame at fps=30, pix_per_mm=10 → 3 mm/s.
    cx = np.arange(n, dtype=float)
    cy = np.full(n, 50.0)
    df = _synth_track_df(cx, cy)
    cleaned = clean_trajectory(
        df, fps=30.0, pix_per_mm=10.0, sg_window=5, sg_polyorder=2,
    )
    # Use the middle slice to dodge SG edge effects.
    middle = cleaned.iloc[20:40]
    assert np.allclose(middle["vx_mm_s"].to_numpy(), 3.0, atol=0.05)
    assert np.allclose(middle["vy_mm_s"].to_numpy(), 0.0, atol=0.05)
    assert np.allclose(middle["speed_mm_s"].to_numpy(), 3.0, atol=0.05)


def test_clean_track_handles_multi_track_input():
    n = 30
    cx_a = np.arange(n, dtype=float)
    cx_b = np.arange(n, dtype=float) + 100.0
    cy = np.full(n, 50.0)
    df_a = _synth_track_df(cx_a, cy, track_id=0)
    df_b = _synth_track_df(cx_b, cy, track_id=1)
    df = pd.concat([df_a, df_b], ignore_index=True)
    cleaned = clean_trajectory(
        df, fps=30.0, pix_per_mm=10.0, sg_window=5, sg_polyorder=2,
    )
    assert sorted(cleaned["track_id"].unique().tolist()) == [0, 1]
    assert len(cleaned[cleaned["track_id"] == 0]) == n
    assert len(cleaned[cleaned["track_id"] == 1]) == n

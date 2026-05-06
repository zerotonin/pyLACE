"""Body-axis disambiguation + yaw + yaw rate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pylace.posthoc.heading import compute_yaw, resolve_headings


def _synth_walker_df(
    cx: np.ndarray, cy: np.ndarray, *,
    orientation_deg: np.ndarray,
    track_id: int = 0,
) -> pd.DataFrame:
    """Build a DataFrame that already has cx_smooth/vx/vy/speed columns
    set, simulating the output of clean_trajectory but cheaper to set up.
    """
    n = cx.size
    dt = 1.0 / 30.0
    vx = np.gradient(cx, dt) * 0.1   # 0.1 mm/px (pix_per_mm=10)
    vy = np.gradient(cy, dt) * 0.1
    speed = np.hypot(vx, vy)
    return pd.DataFrame({
        "frame_idx": np.arange(n),
        "track_id": [track_id] * n,
        "cx_smooth_px": cx,
        "cy_smooth_px": cy,
        "vx_mm_s": vx,
        "vy_mm_s": vy,
        "speed_mm_s": speed,
        "orientation_deg": orientation_deg,
    })


def test_heading_aligns_with_velocity_for_walker():
    """A fly walking +x with body-axis +x should emerge with heading ≈ 0°."""
    n = 60
    cx = np.arange(n, dtype=float) * 5.0
    cy = np.full(n, 50.0)
    # Ellipse orientation reports 0° for a horizontal ellipse, ambiguous
    # between 0° (head right) and 180° (head left). Velocity is +x (~0°).
    orient = np.full(n, 0.0)
    df = _synth_walker_df(cx, cy, orientation_deg=orient)
    out = resolve_headings(df)
    middle = out.iloc[10:50]["heading_deg"].to_numpy()
    diffs = np.minimum(middle, 360.0 - middle)
    assert (diffs < 5.0).all(), f"Expected heading ≈ 0°, got {middle[:5]}"


def test_heading_picks_180_when_walking_left():
    n = 60
    cx = 100.0 - np.arange(n, dtype=float) * 5.0  # walking -x
    cy = np.full(n, 50.0)
    orient = np.full(n, 0.0)  # body axis horizontal, ambiguous
    df = _synth_walker_df(cx, cy, orientation_deg=orient)
    out = resolve_headings(df)
    middle = out.iloc[10:50]["heading_deg"].to_numpy()
    diffs = np.abs(middle - 180.0)
    assert (diffs < 5.0).all(), f"Expected heading ≈ 180°, got {middle[:5]}"


def test_heading_dp_resists_pi_flip_on_brief_stationarity():
    """If the fly stops for a few frames, heading should not flip."""
    n = 60
    cx = np.zeros(n)
    cy = np.zeros(n)
    cx[:20] = np.arange(20) * 5.0          # walking +x
    cx[20:30] = cx[19]                      # stop for 10 frames
    cx[30:] = cx[29] + np.arange(n - 30) * 5.0  # resume +x
    orient = np.full(n, 0.0)
    df = _synth_walker_df(cx, cy, orientation_deg=orient)
    out = resolve_headings(df)
    # No flip across the stationary window.
    headings = out["heading_deg"].to_numpy()
    diffs = np.minimum(headings, 360.0 - headings)
    assert (diffs < 5.0).all()


def test_heading_unresolved_flag_when_orientation_is_nan():
    n = 30
    cx = np.arange(n, dtype=float)
    cy = np.full(n, 50.0)
    orient = np.full(n, 0.0)
    orient[10:15] = np.nan  # missing orientation for 5 frames
    df = _synth_walker_df(cx, cy, orientation_deg=orient)
    out = resolve_headings(df)
    assert out.loc[10:14, "heading_unresolved"].all()
    assert not out.loc[:9, "heading_unresolved"].any()


def test_yaw_unwraps_continuous_rotation():
    """A fly turning steadily through 360°+ should produce a monotonic yaw."""
    n = 100
    # Heading rotates 720° over the recording.
    heading = (np.linspace(0, 720, n)) % 360.0
    df = pd.DataFrame({
        "frame_idx": np.arange(n),
        "track_id": [0] * n,
        "heading_deg": heading,
    })
    out = compute_yaw(df, fps=30.0, sg_window=5, sg_polyorder=2)
    yaw = out["yaw_deg"].to_numpy()
    # Should be monotonically increasing (after smoothing).
    diffs = np.diff(yaw[3:-3])  # avoid SG edge effects
    assert (diffs > 0).all()
    # Total rotation ≈ 720°.
    assert 700.0 < (yaw[-1] - yaw[0]) < 740.0


def test_yaw_rate_zero_for_constant_heading():
    n = 50
    df = pd.DataFrame({
        "frame_idx": np.arange(n),
        "track_id": [0] * n,
        "heading_deg": np.full(n, 45.0),
    })
    out = compute_yaw(df, fps=30.0, sg_window=5, sg_polyorder=2)
    middle = out.iloc[10:40]["yaw_rate_deg_s"].to_numpy()
    assert np.allclose(middle, 0.0, atol=1e-6)


def test_yaw_rate_matches_known_constant_turning_rate():
    """A fly turning at 90°/s for 2 s should produce yaw_rate ≈ 90 °/s."""
    n = 60       # 2 s at 30 fps
    fps = 30.0
    heading = (np.arange(n) * (90.0 / fps)) % 360.0
    df = pd.DataFrame({
        "frame_idx": np.arange(n),
        "track_id": [0] * n,
        "heading_deg": heading,
    })
    out = compute_yaw(df, fps=fps, sg_window=5, sg_polyorder=2)
    middle = out.iloc[10:50]["yaw_rate_deg_s"].to_numpy()
    assert np.allclose(middle, 90.0, atol=1.0)

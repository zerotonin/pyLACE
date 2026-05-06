# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — posthoc.heading                                        ║
# ║  « head/tail disambiguation + yaw + yaw rate »                   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  cv2.fitEllipse returns orientation modulo π, so the body axis   ║
# ║  is geometrically ambiguous. The two facts that resolve it:      ║
# ║    1. Walking flies almost always travel forward, so the head    ║
# ║       end aligns with the velocity vector when speed is high.    ║
# ║    2. Heading rarely changes by more than ~90° between adjacent  ║
# ║       frames, so a flip relative to the previous frame is        ║
# ║       evidence against itself.                                   ║
# ║  A two-state Viterbi over the (θ, θ+180°) candidates per frame   ║
# ║  combines both, with a tunable transition penalty for π flips.   ║
# ║  Stationary frames have zero unary cost; the transition cost     ║
# ║  alone keeps heading consistent across them.                     ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Body-axis disambiguation + yaw + yaw rate from cleaned trajectories."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from pylace.posthoc.constants import (
    DEFAULT_HEADING_FLIP_PENALTY,
    DEFAULT_HEADING_SPEED_FLOOR_MM_S,
    DEFAULT_YAW_SG_POLYORDER,
    DEFAULT_YAW_SG_WINDOW,
)


def resolve_headings(
    df: pd.DataFrame,
    *,
    speed_floor_mm_s: float = DEFAULT_HEADING_SPEED_FLOOR_MM_S,
    flip_penalty: float = DEFAULT_HEADING_FLIP_PENALTY,
) -> pd.DataFrame:
    """Resolve head/tail per track via velocity-anchored Viterbi.

    Args:
        df: Cleaned trajectory DataFrame with ``orientation_deg``,
            ``vx_mm_s``, ``vy_mm_s``, ``speed_mm_s``, ``track_id``.
        speed_floor_mm_s: Below this, the velocity vector is too
            small to anchor head/tail. The DP step still chooses, but
            the unary cost is set to zero so adjacent frames win.
        flip_penalty: Cost (degrees-equivalent) of a ~180° flip
            between adjacent frames. Higher → smoother heading.

    Returns:
        Same DataFrame with a new ``heading_deg`` column in
        [0, 360) and a boolean ``heading_unresolved`` column flagging
        frames whose orientation was missing in the input.
    """
    out = df.copy()
    headings = np.full(len(out), np.nan)
    unresolved = np.ones(len(out), dtype=bool)
    for tid, group in out.groupby("track_id", sort=False):
        idx = group.index.to_numpy()
        h, ur = _resolve_one_track(
            group, speed_floor_mm_s=speed_floor_mm_s,
            flip_penalty=flip_penalty,
        )
        headings[idx] = h
        unresolved[idx] = ur
    out["heading_deg"] = headings
    out["heading_unresolved"] = unresolved
    return out


def compute_yaw(
    df: pd.DataFrame,
    *,
    fps: float,
    sg_window: int = DEFAULT_YAW_SG_WINDOW,
    sg_polyorder: int = DEFAULT_YAW_SG_POLYORDER,
) -> pd.DataFrame:
    """Add ``yaw_deg`` (unwrapped) and ``yaw_rate_deg_s`` to the trajectory.

    Smooths ``cos`` and ``sin`` of the resolved heading separately
    with a small Savitzky-Golay window and recovers a continuous
    yaw via ``atan2`` + ``np.unwrap``. The rate is the time derivative
    of the unwrapped yaw.

    Args:
        df: Trajectory with ``heading_deg`` from ``resolve_headings``.
        fps: Source frame rate (used for the rate scaling).
        sg_window: SG window for the angle-component smoother.
        sg_polyorder: SG polynomial order.
    """
    out = df.copy()
    yaw_deg = np.full(len(out), np.nan)
    yaw_rate = np.full(len(out), np.nan)
    for _, group in out.groupby("track_id", sort=False):
        idx = group.index.to_numpy()
        y, r = _yaw_one_track(
            group["heading_deg"].to_numpy(),
            fps=fps, sg_window=sg_window, sg_polyorder=sg_polyorder,
        )
        yaw_deg[idx] = y
        yaw_rate[idx] = r
    out["yaw_deg"] = yaw_deg
    out["yaw_rate_deg_s"] = yaw_rate
    return out


# ─────────────────────────────────────────────────────────────────
#  Per-track Viterbi
# ─────────────────────────────────────────────────────────────────


def _resolve_one_track(
    group: pd.DataFrame,
    *,
    speed_floor_mm_s: float,
    flip_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta = group["orientation_deg"].to_numpy(dtype=float)
    vx = group["vx_mm_s"].to_numpy(dtype=float)
    vy = group["vy_mm_s"].to_numpy(dtype=float)
    speed = group["speed_mm_s"].to_numpy(dtype=float)
    n = theta.size
    unresolved = np.isnan(theta)
    if n == 0:
        return np.full(0, np.nan), np.zeros(0, dtype=bool)

    # Two heading candidates per frame: θ and θ+180.
    cand0 = theta % 360.0
    cand1 = (theta + 180.0) % 360.0

    # Velocity direction in degrees (0 = +x, 90 = +y).
    vel_deg = np.degrees(np.arctan2(vy, vx)) % 360.0
    vel_known = (~np.isnan(speed)) & (speed >= speed_floor_mm_s)

    # Unary cost = circular distance to velocity bearing if known, else 0.
    unary0 = np.where(vel_known, _circ_dist(cand0, vel_deg), 0.0)
    unary1 = np.where(vel_known, _circ_dist(cand1, vel_deg), 0.0)

    # Run a 2-state Viterbi over present-θ frames; copy through gaps.
    dp = np.full((n, 2), np.inf)
    back = np.zeros((n, 2), dtype=np.int8)

    first_present = -1
    for i in range(n):
        if unresolved[i]:
            continue
        if first_present < 0:
            dp[i, 0] = unary0[i]
            dp[i, 1] = unary1[i]
            first_present = i
            continue
        prev = _last_present_index(unresolved, i)
        if prev < 0:
            dp[i, 0] = unary0[i]
            dp[i, 1] = unary1[i]
            continue
        for k in (0, 1):
            cand_k = cand0[i] if k == 0 else cand1[i]
            best_cost = np.inf
            best_prev = 0
            for kp in (0, 1):
                cand_kp = cand0[prev] if kp == 0 else cand1[prev]
                trans = _transition_cost(cand_kp, cand_k, flip_penalty)
                c = dp[prev, kp] + trans
                if c < best_cost:
                    best_cost = c
                    best_prev = kp
            unary = unary0[i] if k == 0 else unary1[i]
            dp[i, k] = best_cost + unary
            back[i, k] = best_prev

    if first_present < 0:
        return np.full(n, np.nan), unresolved

    # Backtrace from the last present frame.
    last_present = _last_present_index(unresolved, n)
    if last_present < 0:
        return np.full(n, np.nan), unresolved
    state = int(np.argmin(dp[last_present]))
    chosen = np.full(n, -1, dtype=np.int8)
    chosen[last_present] = state
    cur_state = state
    cur = last_present
    while cur > first_present:
        prev = _last_present_index(unresolved, cur)
        if prev < 0:
            break
        cur_state = int(back[cur, cur_state])
        chosen[prev] = cur_state
        cur = prev

    headings = np.full(n, np.nan)
    for i in range(n):
        if chosen[i] >= 0:
            headings[i] = cand0[i] if chosen[i] == 0 else cand1[i]
    # Forward/back-fill across NaN gaps so unresolved frames get the
    # nearest known heading rather than NaN — matches the "stationary
    # fly inherits previous orientation" intent.
    headings = _ffill_bfill(headings)
    return headings, unresolved


def _yaw_one_track(
    heading_deg: np.ndarray,
    *,
    fps: float,
    sg_window: int,
    sg_polyorder: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = heading_deg.size
    if n == 0:
        return np.zeros(0), np.zeros(0)
    theta = np.deg2rad(heading_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    valid = ~np.isnan(heading_deg)
    if valid.sum() < max(2, sg_window):
        # Too few points to smooth; fall back to raw unwrap.
        yaw_unwrapped = np.full(n, np.nan)
        yaw_rate = np.full(n, np.nan)
        if valid.sum() >= 2:
            yaw_unwrapped[valid] = np.degrees(np.unwrap(theta[valid]))
            yaw_rate[valid] = np.gradient(yaw_unwrapped[valid], 1.0 / fps)
        return yaw_unwrapped, yaw_rate

    # Smooth cos/sin separately, then recover heading via atan2.
    cos_s = cos_t.copy()
    sin_s = sin_t.copy()
    cos_s[valid] = savgol_filter(
        cos_t[valid], window_length=sg_window,
        polyorder=sg_polyorder, mode="interp",
    )
    sin_s[valid] = savgol_filter(
        sin_t[valid], window_length=sg_window,
        polyorder=sg_polyorder, mode="interp",
    )
    smoothed = np.arctan2(sin_s, cos_s)
    yaw_unwrapped = np.full(n, np.nan)
    yaw_rate = np.full(n, np.nan)
    yaw_unwrapped[valid] = np.degrees(np.unwrap(smoothed[valid]))
    yaw_rate[valid] = np.gradient(yaw_unwrapped[valid], 1.0 / fps)
    return yaw_unwrapped, yaw_rate


# ─────────────────────────────────────────────────────────────────
#  Geometry + array helpers
# ─────────────────────────────────────────────────────────────────


def _circ_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest absolute angular difference in degrees, in [0, 180]."""
    d = (a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def _transition_cost(prev_deg: float, cur_deg: float, flip_penalty: float) -> float:
    """Penalise transitions whose angular jump exceeds 90°.

    Below 90° the jump is plausible fly motion and costs nothing; above
    90° it is interpreted as a head/tail flip and costs scale linearly
    with the angular jump up to a maximum of ``flip_penalty`` at 180°.
    """
    d = abs((prev_deg - cur_deg + 180.0) % 360.0 - 180.0)
    if d <= 90.0:
        return 0.0
    return flip_penalty * (d - 90.0) / 90.0


def _last_present_index(unresolved: np.ndarray, before: int) -> int:
    """Index of the most recent frame ``< before`` whose orientation is known."""
    for k in range(before - 1, -1, -1):
        if not unresolved[k]:
            return k
    return -1


def _ffill_bfill(arr: np.ndarray) -> np.ndarray:
    """Forward-fill then back-fill NaNs in a 1-D float array."""
    out = arr.astype(float, copy=True)
    n = out.size
    if n == 0:
        return out
    # Forward fill.
    last = np.nan
    for i in range(n):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    # Back fill leading NaNs.
    last = np.nan
    for i in range(n - 1, -1, -1):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out

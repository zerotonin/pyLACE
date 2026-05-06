# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — posthoc.clean                                          ║
# ║  « gap fill → outlier rejection → Savitzky-Golay smoothing »     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  The first three stages of the cleaner. Heading + yaw live in    ║
# ║  posthoc.heading and consume the smoothed positions and          ║
# ║  velocities this module produces.                                ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Trajectory-level cleaning: gap fill, outlier rejection, SG smoothing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from pylace.posthoc.constants import (
    DEFAULT_MAX_GAP_FRACTION_OF_FPS,
    DEFAULT_MAX_SPEED_MM_S,
    DEFAULT_SG_POLYORDER,
    DEFAULT_SG_WINDOW,
)


def clean_trajectory(
    df: pd.DataFrame,
    *,
    fps: float,
    pix_per_mm: float,
    max_gap_frames: int | None = None,
    max_speed_mm_s: float = DEFAULT_MAX_SPEED_MM_S,
    sg_window: int = DEFAULT_SG_WINDOW,
    sg_polyorder: int = DEFAULT_SG_POLYORDER,
) -> pd.DataFrame:
    """Clean a multi-track detection DataFrame in place of a per-track loop.

    Pipeline per track: reindex to the contiguous frame range,
    velocity-threshold outlier rejection on the raw observations,
    linear interpolation of small gaps, Savitzky-Golay smoothing of
    the position columns, and finite-difference velocity from the
    smoothed positions.

    Args:
        df: Detection DataFrame from ``pylace.posthoc.io.read_detections``.
        fps: Source frame rate.
        pix_per_mm: Pixel-to-millimetre conversion (from sidecar).
        max_gap_frames: Largest gap (in frames) we will linearly
            interpolate. Defaults to ``round(fps / 12)`` (~1/12 s).
            Gaps beyond this are left as missing rows.
        max_speed_mm_s: Velocity threshold for single-frame teleport
            rejection.
        sg_window: Savitzky-Golay window length (odd, > polyorder).
        sg_polyorder: Savitzky-Golay polynomial order.

    Returns:
        New DataFrame with original columns plus ``cx_smooth_px``,
        ``cy_smooth_px``, ``vx_mm_s``, ``vy_mm_s``, ``speed_mm_s``,
        ``interpolated``, ``outlier_rejected``. Heading-related
        columns are added downstream by ``resolve_headings`` and
        ``compute_yaw``.
    """
    if pix_per_mm <= 0:
        raise ValueError(f"pix_per_mm must be positive, got {pix_per_mm}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if max_gap_frames is None:
        max_gap_frames = max(1, int(round(fps * DEFAULT_MAX_GAP_FRACTION_OF_FPS)))

    cleaned: list[pd.DataFrame] = []
    for _, group in df.groupby("track_id", sort=True):
        cleaned.append(
            _clean_one_track(
                group,
                fps=fps, pix_per_mm=pix_per_mm,
                max_gap_frames=max_gap_frames,
                max_speed_mm_s=max_speed_mm_s,
                sg_window=sg_window,
                sg_polyorder=sg_polyorder,
            ),
        )
    return pd.concat(cleaned, ignore_index=True)


def _clean_one_track(
    group: pd.DataFrame,
    *,
    fps: float,
    pix_per_mm: float,
    max_gap_frames: int,
    max_speed_mm_s: float,
    sg_window: int,
    sg_polyorder: int,
) -> pd.DataFrame:
    g = (
        group.sort_values("frame_idx").reset_index(drop=True).copy()
    )
    full_idx = np.arange(int(g["frame_idx"].iloc[0]),
                         int(g["frame_idx"].iloc[-1]) + 1)
    g_full = g.set_index("frame_idx").reindex(full_idx)
    g_full["interpolated"] = g_full["cx_px"].isna()
    g_full["outlier_rejected"] = False

    # Outlier rejection on the raw observations only — interpolated
    # positions are not yet present.
    g_full = _flag_outliers(
        g_full, fps=fps, pix_per_mm=pix_per_mm,
        max_speed_mm_s=max_speed_mm_s,
    )

    g_full["cx_px"] = _interpolate_with_max_gap(
        g_full["cx_px"].to_numpy(), max_gap_frames,
    )
    g_full["cy_px"] = _interpolate_with_max_gap(
        g_full["cy_px"].to_numpy(), max_gap_frames,
    )

    cx_smooth = _sg_smooth_runs(
        g_full["cx_px"].to_numpy(), sg_window, sg_polyorder,
    )
    cy_smooth = _sg_smooth_runs(
        g_full["cy_px"].to_numpy(), sg_window, sg_polyorder,
    )
    g_full["cx_smooth_px"] = cx_smooth
    g_full["cy_smooth_px"] = cy_smooth

    dt = 1.0 / fps
    g_full["vx_mm_s"] = _safe_gradient(cx_smooth, dt) / pix_per_mm
    g_full["vy_mm_s"] = _safe_gradient(cy_smooth, dt) / pix_per_mm
    g_full["speed_mm_s"] = np.hypot(g_full["vx_mm_s"], g_full["vy_mm_s"])

    # track_id and roi_label survive the reindex by ffill (they're
    # constant within a track).
    g_full["track_id"] = group["track_id"].iloc[0]
    if "roi_label" in g_full.columns:
        g_full["roi_label"] = (
            g_full["roi_label"].ffill().bfill()
        )

    g_full = g_full.reset_index().rename(columns={"index": "frame_idx"})
    return g_full


# ─────────────────────────────────────────────────────────────────
#  Stage helpers
# ─────────────────────────────────────────────────────────────────


def _flag_outliers(
    g_full: pd.DataFrame,
    *,
    fps: float,
    pix_per_mm: float,
    max_speed_mm_s: float,
) -> pd.DataFrame:
    """Mark single-frame teleports as ``outlier_rejected`` and NaN them."""
    cx = g_full["cx_px"].to_numpy(dtype=float, copy=True)
    cy = g_full["cy_px"].to_numpy(dtype=float, copy=True)
    n = cx.size
    valid = np.where(~np.isnan(cx))[0]
    rejected = np.zeros(n, dtype=bool)
    for j in range(1, valid.size):
        i = int(valid[j])
        prev = int(valid[j - 1])
        dt_frames = i - prev
        if dt_frames <= 0:
            continue
        dx = cx[i] - cx[prev]
        dy = cy[i] - cy[prev]
        speed_mm_s = (np.hypot(dx, dy) / pix_per_mm) * (fps / dt_frames)
        if speed_mm_s > max_speed_mm_s:
            rejected[i] = True
            cx[i] = np.nan
            cy[i] = np.nan
    g_full["cx_px"] = cx
    g_full["cy_px"] = cy
    g_full["outlier_rejected"] = (
        g_full["outlier_rejected"].to_numpy() | rejected
    )
    return g_full


def _interpolate_with_max_gap(
    arr: np.ndarray, max_gap: int,
) -> np.ndarray:
    """Linearly interpolate NaN runs of length ≤ max_gap; longer runs stay NaN."""
    out = arr.astype(float, copy=True)
    n = out.size
    if n == 0:
        return out
    isnan = np.isnan(out)
    if not isnan.any():
        return out

    # Walk the NaN runs.
    i = 0
    while i < n:
        if not isnan[i]:
            i += 1
            continue
        j = i
        while j < n and isnan[j]:
            j += 1
        gap_len = j - i
        # Need anchors on both sides of the gap.
        if i > 0 and j < n and gap_len <= max_gap:
            x0, x1 = i - 1, j
            y0, y1 = out[x0], out[x1]
            for k in range(i, j):
                t = (k - x0) / (x1 - x0)
                out[k] = y0 + t * (y1 - y0)
        i = j
    return out


def _sg_smooth_runs(
    arr: np.ndarray, window: int, polyorder: int,
) -> np.ndarray:
    """Apply Savitzky-Golay to each contiguous non-NaN run; short runs untouched."""
    out = arr.astype(float, copy=True)
    if window <= polyorder:
        raise ValueError(
            f"sg_window ({window}) must be greater than sg_polyorder ({polyorder})",
        )
    if window % 2 == 0:
        raise ValueError(f"sg_window must be odd, got {window}")
    n = out.size
    if n == 0:
        return out
    isnan = np.isnan(out)
    i = 0
    while i < n:
        if isnan[i]:
            i += 1
            continue
        j = i
        while j < n and not isnan[j]:
            j += 1
        run = out[i:j]
        if run.size >= window:
            out[i:j] = savgol_filter(
                run, window_length=window, polyorder=polyorder, mode="interp",
            )
        i = j
    return out


def _safe_gradient(arr: np.ndarray, dt: float) -> np.ndarray:
    """``np.gradient`` that propagates NaN through the central-difference stencil."""
    out = np.full_like(arr, np.nan, dtype=float)
    if arr.size < 2:
        return out
    valid = ~np.isnan(arr)
    if valid.sum() < 2:
        return out
    # Compute gradient over each contiguous valid run separately.
    n = arr.size
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        if j - i >= 2:
            out[i:j] = np.gradient(arr[i:j], dt)
        i = j
    return out

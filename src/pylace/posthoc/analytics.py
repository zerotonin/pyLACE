# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — posthoc.analytics                                      ║
# ║  « derived columns from a cleaned trajectory + arena geometry »  ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Derived time series from a single-track cleaned trajectory."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from pylace.annotator.geometry import Arena, Circle


def compute_distance_to_wall(
    traj_track: pd.DataFrame, arena: Arena, *, pix_per_mm: float,
) -> np.ndarray:
    """Distance from each centroid to the nearest arena edge, in millimetres.

    Positive when the centroid is inside the arena; negative when it
    has drifted outside (which the per-frame mask guard usually
    prevents, but we leave the sign meaningful).

    Args:
        traj_track: Single-track cleaned trajectory with
            ``cx_smooth_px`` and ``cy_smooth_px``.
        arena: Arena geometry from the sidecar.
        pix_per_mm: Pixel-to-millimetre conversion.

    Returns:
        ``(N,)`` float64 array of signed distance (mm), NaN where the
        smoothed centroid is missing.
    """
    if pix_per_mm <= 0:
        raise ValueError(f"pix_per_mm must be positive, got {pix_per_mm}")
    cx = traj_track["cx_smooth_px"].to_numpy(dtype=float)
    cy = traj_track["cy_smooth_px"].to_numpy(dtype=float)
    n = cx.size
    out = np.full(n, np.nan, dtype=float)
    valid = ~(np.isnan(cx) | np.isnan(cy))
    if not valid.any():
        return out
    if isinstance(arena, Circle):
        d_centre = np.hypot(cx[valid] - arena.cx, cy[valid] - arena.cy)
        out[valid] = (arena.r - d_centre) / pix_per_mm
        return out
    contour = np.asarray(arena.vertices, dtype=np.float32).reshape(-1, 1, 2)
    valid_idx = np.where(valid)[0]
    for i in valid_idx:
        signed_px = cv2.pointPolygonTest(
            contour, (float(cx[i]), float(cy[i])), True,
        )
        out[i] = signed_px / pix_per_mm
    return out

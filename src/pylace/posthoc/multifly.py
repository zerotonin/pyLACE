# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — posthoc.multifly                                       ║
# ║  « pairwise distances, nearest neighbour, polarisation »         ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Multi-track readouts that only make sense when the recording    ║
# ║  has more than one fly. Conventions:                             ║
# ║                                                                  ║
# ║    pairwise_distance_matrix → (T, N, N) tensor in mm, with       ║
# ║                              tracks ordered by id.               ║
# ║    nearest_neighbour_distance → long-form (frame_idx, track_id, ║
# ║                              nn_distance_mm) DataFrame.          ║
# ║    polarisation             → per-frame order parameter          ║
# ║                              r = |⟨ exp(i θ_k) ⟩| over flies     ║
# ║                              with a known heading. r ∈ [0, 1];   ║
# ║                              informative at N ≥ 3 (Couzin et     ║
# ║                              al., Nature 2005).                  ║
# ║                                                                  ║
# ║  All three operate on the cleaned-trajectory DataFrame produced  ║
# ║  by ``pylace-clean`` (must contain cx_smooth_px, cy_smooth_px,   ║
# ║  track_id, frame_idx, and — for polarisation — heading_deg).     ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Multi-fly time-series metrics from a cleaned trajectory."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pairwise_distance_matrix(
    traj: pd.DataFrame, *, pix_per_mm: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return ``(distances_mm, frame_idx, track_ids)``.

    ``distances_mm`` is a ``(T, N, N)`` symmetric matrix where
    ``[t, i, j]`` is the centre-to-centre distance between track i and
    track j at frame index ``frame_idx[t]``. Diagonal entries are
    zero. Frames where either track has a missing centroid yield
    NaN for the corresponding pair.

    Args:
        traj: Cleaned multi-track trajectory.
        pix_per_mm: Pixel-to-millimetre conversion (from sidecar).

    Returns:
        ``(distances_mm, frame_idx, track_ids)`` where
        ``frame_idx`` is the ``T``-length vector of frame indices and
        ``track_ids`` is the ``N``-length list ordered ascending.
    """
    if pix_per_mm <= 0:
        raise ValueError(f"pix_per_mm must be positive, got {pix_per_mm}")
    pivot_x = traj.pivot(
        index="frame_idx", columns="track_id", values="cx_smooth_px",
    ).sort_index().sort_index(axis=1)
    pivot_y = traj.pivot(
        index="frame_idx", columns="track_id", values="cy_smooth_px",
    ).sort_index().sort_index(axis=1)
    cx = pivot_x.to_numpy()       # (T, N)
    cy = pivot_y.to_numpy()
    dx = cx[:, :, None] - cx[:, None, :]   # (T, N, N)
    dy = cy[:, :, None] - cy[:, None, :]
    distances_px = np.hypot(dx, dy)
    distances_mm = distances_px / pix_per_mm
    return (
        distances_mm,
        pivot_x.index.to_numpy(dtype=np.int64),
        [int(c) for c in pivot_x.columns],
    )


def nearest_neighbour_distance(
    traj: pd.DataFrame, *, pix_per_mm: float,
) -> pd.DataFrame:
    """Long-form (frame_idx, track_id, nn_distance_mm) DataFrame.

    For each frame and each track, records the distance in
    millimetres to the nearest *other* track. NaN where the track
    or its nearest neighbour has no centroid that frame.

    Schneider & Levine, *eLife* 2014, used this as the foundational
    social-behaviour readout.
    """
    d_mm, frame_idx, track_ids = pairwise_distance_matrix(
        traj, pix_per_mm=pix_per_mm,
    )
    n = len(track_ids)
    if n < 2:
        return pd.DataFrame(
            columns=["frame_idx", "track_id", "nn_distance_mm"],
        )
    mask_self = np.eye(n, dtype=bool)
    d_no_self = np.where(mask_self[None, :, :], np.inf, d_mm)
    # Replace inf with NaN where the row has only NaN partners.
    all_nan_partners = np.all(
        np.isnan(d_mm) | mask_self[None, :, :], axis=2,
    )
    nn = np.nanmin(d_no_self, axis=2)
    nn[all_nan_partners] = np.nan
    nn_wide = pd.DataFrame(nn, index=frame_idx, columns=track_ids)
    nn_wide.index.name = "frame_idx"
    nn_wide.columns.name = "track_id"
    long = nn_wide.stack(future_stack=True).reset_index(name="nn_distance_mm")
    return long


def polarisation(traj: pd.DataFrame) -> pd.DataFrame:
    """Per-frame order parameter ``r = |⟨ exp(iθ_k) ⟩|`` in ``[0, 1]``.

    The mean is taken over flies whose ``heading_deg`` is known at
    that frame; ``n_valid_tracks`` reports how many contributed. When
    fewer than two flies have a heading, the order parameter is NaN
    (a single bearing aligns trivially with itself; not informative).

    Couzin et al., *Nature* 2005.
    """
    pivot_h = traj.pivot(
        index="frame_idx", columns="track_id", values="heading_deg",
    ).sort_index().sort_index(axis=1)
    h_deg = pivot_h.to_numpy()                 # (T, N)
    valid = ~np.isnan(h_deg)
    h_rad = np.where(valid, np.deg2rad(h_deg), 0.0)
    e_ix = np.where(valid, np.exp(1j * h_rad), 0.0 + 0.0j)
    n_valid = valid.sum(axis=1)
    summed = e_ix.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        order = np.abs(summed) / n_valid
    order[n_valid < 2] = np.nan
    return pd.DataFrame({
        "frame_idx": pivot_h.index.to_numpy(dtype=np.int64),
        "polarisation": order,
        "n_valid_tracks": n_valid.astype(np.int32),
    })


__all__ = [
    "nearest_neighbour_distance",
    "pairwise_distance_matrix",
    "polarisation",
]

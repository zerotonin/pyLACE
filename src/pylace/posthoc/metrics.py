# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — posthoc.metrics                                        ║
# ║  « kinematic readouts: speed, bouts, occupancy, thigmotaxis »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Per-track summaries and 2D distributions computed from the      ║
# ║  cleaned trajectory CSV. The conventions follow the Branson /    ║
# ║  Robie / Anderson lineage of Drosophila open-field assays:       ║
# ║                                                                  ║
# ║    speed_summary      — median, p90, max, total path length      ║
# ║    walk_stop_bouts    — Schmitt-trigger walk / stop bouts        ║
# ║    occupancy_heatmap  — 2D probability density of position       ║
# ║    thigmotaxis_fraction — fraction of time in the outer band     ║
# ║    yaw_rate_summary   — median / 95th-pct of |yaw rate|          ║
# ║                                                                  ║
# ║  summarise_track / summarise_tracks compose them into a single   ║
# ║  one-row-per-fly Pandas DataFrame ready for reRandomStats.       ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Per-track kinematic readouts."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from pylace.annotator.geometry import Arena, Circle
from pylace.posthoc.constants import (
    DEFAULT_BOUT_MIN_DURATION_S,
    DEFAULT_BOUT_OFF_MM_S,
    DEFAULT_BOUT_ON_MM_S,
    DEFAULT_OCCUPANCY_BINS,
    DEFAULT_THIGMOTAXIS_OUTER_FRAC,
)


# ─────────────────────────────────────────────────────────────────
#  Speed
# ─────────────────────────────────────────────────────────────────


def speed_summary(traj_track: pd.DataFrame, *, fps: float) -> dict[str, float]:
    """Median / p90 / max instantaneous speed, plus integrated path length."""
    s = traj_track["speed_mm_s"].to_numpy(dtype=float)
    s = s[~np.isnan(s)]
    if s.size == 0:
        return {
            "speed_median_mm_s": float("nan"),
            "speed_p90_mm_s": float("nan"),
            "speed_max_mm_s": float("nan"),
            "path_length_mm": float("nan"),
        }
    return {
        "speed_median_mm_s": float(np.median(s)),
        "speed_p90_mm_s": float(np.percentile(s, 90)),
        "speed_max_mm_s": float(np.max(s)),
        "path_length_mm": float(s.sum() / fps),
    }


# ─────────────────────────────────────────────────────────────────
#  Walk / stop bouts (Schmitt trigger)
# ─────────────────────────────────────────────────────────────────


def walk_stop_bouts(
    traj_track: pd.DataFrame,
    *,
    fps: float,
    on_threshold_mm_s: float = DEFAULT_BOUT_ON_MM_S,
    off_threshold_mm_s: float = DEFAULT_BOUT_OFF_MM_S,
    min_duration_s: float = DEFAULT_BOUT_MIN_DURATION_S,
) -> pd.DataFrame:
    """Schmitt-trigger classifier; returns one row per walking bout.

    Args:
        traj_track: Single-track cleaned trajectory.
        fps: Frame rate (for converting frame counts to seconds).
        on_threshold_mm_s: Speed above which "stopped" → "walking".
        off_threshold_mm_s: Speed below which "walking" → "stopped".
            Hysteresis (on > off) prevents bout fragmentation.
        min_duration_s: Walks shorter than this are dropped as noise.

    Returns:
        DataFrame with columns ``start_frame``, ``end_frame``,
        ``duration_s``, ``mean_speed_mm_s``. Empty if the fly never
        crosses the on threshold for long enough.
    """
    if on_threshold_mm_s < off_threshold_mm_s:
        raise ValueError(
            f"on_threshold ({on_threshold_mm_s}) must be >= off_threshold "
            f"({off_threshold_mm_s})",
        )
    speed = traj_track["speed_mm_s"].to_numpy(dtype=float)
    n = speed.size
    is_walking = np.zeros(n, dtype=bool)
    state = False
    for i in range(n):
        v = speed[i]
        if np.isnan(v):
            is_walking[i] = state
            continue
        if state and v < off_threshold_mm_s:
            state = False
        elif (not state) and v > on_threshold_mm_s:
            state = True
        is_walking[i] = state

    rows: list[tuple[int, int, float, float]] = []
    frame_idx = traj_track["frame_idx"].to_numpy()
    i = 0
    while i < n:
        if not is_walking[i]:
            i += 1
            continue
        j = i
        while j < n and is_walking[j]:
            j += 1
        duration_s = (j - i) / fps
        if duration_s >= min_duration_s:
            mean_speed = float(np.nanmean(speed[i:j]))
            rows.append(
                (int(frame_idx[i]), int(frame_idx[j - 1]),
                 float(duration_s), mean_speed),
            )
        i = j
    return pd.DataFrame(
        rows,
        columns=["start_frame", "end_frame", "duration_s", "mean_speed_mm_s"],
    )


# ─────────────────────────────────────────────────────────────────
#  Occupancy heatmap
# ─────────────────────────────────────────────────────────────────


def occupancy_heatmap(
    traj_track: pd.DataFrame,
    *,
    frame_size: tuple[int, int],
    n_bins: int = DEFAULT_OCCUPANCY_BINS,
) -> np.ndarray:
    """2D probability density of position over the frame.

    Bins are uniform in pixels across the full frame. The returned
    array is normalised to sum to 1 (per-bin probability mass).
    Reviewers comparing centre-vs-edge time should remember that the
    density is per *bin*, not per-area-of-real-arena — outside the
    arena bins are zero by construction so the density is already
    arena-confined.

    Args:
        traj_track: Single-track cleaned trajectory.
        frame_size: ``(width, height)`` of the source video.
        n_bins: Resolution of the histogram in each axis.

    Returns:
        ``(n_bins, n_bins)`` float64 array; row index = y, column = x.
    """
    cx = traj_track["cx_smooth_px"].to_numpy(dtype=float)
    cy = traj_track["cy_smooth_px"].to_numpy(dtype=float)
    valid = ~(np.isnan(cx) | np.isnan(cy))
    cx = cx[valid]
    cy = cy[valid]
    width, height = frame_size
    if cx.size == 0:
        return np.zeros((n_bins, n_bins), dtype=float)
    h, _, _ = np.histogram2d(
        cy, cx,
        bins=n_bins,
        range=[[0, height], [0, width]],
    )
    total = h.sum()
    return h / total if total > 0 else h


# ─────────────────────────────────────────────────────────────────
#  Thigmotaxis (wall bias)
# ─────────────────────────────────────────────────────────────────


def thigmotaxis_fraction(
    traj_track: pd.DataFrame,
    arena: Arena,
    *,
    outer_band_frac: float = DEFAULT_THIGMOTAXIS_OUTER_FRAC,
) -> float:
    """Fraction of valid frames spent within the outer band of the arena.

    For circular arenas the band is the annulus between
    ``(1 - outer_band_frac) * r`` and ``r``. For rectangles and
    polygons we use the inscribed-circle radius (minimum distance
    from centroid to a vertex) as the effective radius — a simple
    shape-agnostic generalisation that gives a comparable readout
    across arena geometries.

    Args:
        traj_track: Single-track cleaned trajectory.
        arena: Arena geometry from the sidecar.
        outer_band_frac: Width of the outer band, as a fraction of
            the arena radius (default 0.20 → outermost 20%).

    Returns:
        Fraction in [0, 1], or NaN if there are no valid positions.
    """
    cx = traj_track["cx_smooth_px"].to_numpy(dtype=float)
    cy = traj_track["cy_smooth_px"].to_numpy(dtype=float)
    valid = ~(np.isnan(cx) | np.isnan(cy))
    cx = cx[valid]
    cy = cy[valid]
    if cx.size == 0:
        return float("nan")
    a_cx, a_cy, a_r = _arena_centre_and_inscribed_radius(arena)
    d = np.hypot(cx - a_cx, cy - a_cy)
    threshold = a_r * (1.0 - outer_band_frac)
    return float((d > threshold).mean())


def _arena_centre_and_inscribed_radius(arena: Arena) -> tuple[float, float, float]:
    """Centre + true inscribed radius (distance from centre to nearest edge).

    For circles the inscribed radius is the radius itself. For rectangles
    and polygons we use ``cv2.pointPolygonTest`` with ``measureDist=True``
    to get the signed distance from the centroid to the nearest edge, then
    take its absolute value. Distance-to-nearest-vertex would give the
    *circumscribed* radius, which would put the entire arena interior
    "outside" the outer band for any thigmotaxis threshold ≥ 0.
    """
    if isinstance(arena, Circle):
        return float(arena.cx), float(arena.cy), float(arena.r)
    vertices = np.asarray(arena.vertices, dtype=float)
    if vertices.size == 0:
        raise ValueError("Arena has no vertices to compute centre from.")
    cx = float(vertices[:, 0].mean())
    cy = float(vertices[:, 1].mean())
    contour = vertices.astype(np.float32).reshape(-1, 1, 2)
    inscribed = cv2.pointPolygonTest(contour, (cx, cy), True)
    return cx, cy, abs(float(inscribed))


# ─────────────────────────────────────────────────────────────────
#  Yaw rate summary
# ─────────────────────────────────────────────────────────────────


def yaw_rate_summary(traj_track: pd.DataFrame) -> dict[str, float]:
    """Median absolute and 95th-percentile absolute yaw rate (deg/s)."""
    r = traj_track["yaw_rate_deg_s"].to_numpy(dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return {
            "yaw_rate_median_abs_deg_s": float("nan"),
            "yaw_rate_p95_abs_deg_s": float("nan"),
        }
    abs_r = np.abs(r)
    return {
        "yaw_rate_median_abs_deg_s": float(np.median(abs_r)),
        "yaw_rate_p95_abs_deg_s": float(np.percentile(abs_r, 95)),
    }


# ─────────────────────────────────────────────────────────────────
#  Per-track and multi-track aggregators
# ─────────────────────────────────────────────────────────────────


def summarise_track(
    traj_track: pd.DataFrame,
    *,
    fps: float,
    arena: Arena | None = None,
    nn_distances: pd.DataFrame | None = None,
    on_threshold_mm_s: float = DEFAULT_BOUT_ON_MM_S,
    off_threshold_mm_s: float = DEFAULT_BOUT_OFF_MM_S,
    min_duration_s: float = DEFAULT_BOUT_MIN_DURATION_S,
    outer_band_frac: float = DEFAULT_THIGMOTAXIS_OUTER_FRAC,
) -> dict[str, float]:
    """Compose all scalar metrics into one row of summary stats.

    Args:
        traj_track: Single-track cleaned trajectory.
        fps: Frame rate.
        arena: Optional arena geometry; required for thigmotaxis.
        nn_distances: Optional long-form ``(frame_idx, track_id,
            nn_distance_mm)`` DataFrame from
            :func:`pylace.posthoc.multifly.nearest_neighbour_distance`.
            When provided, adds ``nn_mean_mm`` and ``nn_p10_mm``
            (10th percentile — typical close-approach distance).
    """
    tid = int(traj_track["track_id"].iloc[0])
    out: dict[str, float] = {
        "track_id": tid,
        "n_frames": int(len(traj_track)),
        "total_time_s": float(len(traj_track) / fps),
    }
    out.update(speed_summary(traj_track, fps=fps))
    bouts = walk_stop_bouts(
        traj_track, fps=fps,
        on_threshold_mm_s=on_threshold_mm_s,
        off_threshold_mm_s=off_threshold_mm_s,
        min_duration_s=min_duration_s,
    )
    walking_time = float(bouts["duration_s"].sum())
    out["n_walk_bouts"] = int(len(bouts))
    out["walking_time_s"] = walking_time
    out["walking_fraction"] = (
        walking_time / out["total_time_s"] if out["total_time_s"] > 0 else float("nan")
    )
    out.update(yaw_rate_summary(traj_track))
    if arena is not None:
        out["thigmotaxis_fraction"] = thigmotaxis_fraction(
            traj_track, arena, outer_band_frac=outer_band_frac,
        )
    if nn_distances is not None:
        nn_self = nn_distances[nn_distances["track_id"] == tid]["nn_distance_mm"]
        nn_self = nn_self.dropna()
        if not nn_self.empty:
            out["nn_mean_mm"] = float(nn_self.mean())
            out["nn_p10_mm"] = float(np.percentile(nn_self, 10))
        else:
            out["nn_mean_mm"] = float("nan")
            out["nn_p10_mm"] = float("nan")
    return out


def summarise_tracks(
    traj: pd.DataFrame,
    *,
    fps: float,
    arena: Arena | None = None,
    nn_distances: pd.DataFrame | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Summary frame with one row per track, columns from ``summarise_track``."""
    rows = [
        summarise_track(
            group, fps=fps, arena=arena,
            nn_distances=nn_distances, **kwargs,
        )
        for _, group in traj.groupby("track_id", sort=True)
    ]
    return pd.DataFrame(rows)

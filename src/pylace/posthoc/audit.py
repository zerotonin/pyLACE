# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — posthoc.audit                                          ║
# ║  « sliding-window identity audit on motion-residual cost »       ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Phase 6 lite of the post-hoc roadmap. The Hungarian tracker     ║
# ║  in pyLACE makes assignments one frame at a time; when two       ║
# ║  flies merge into a chain blob and re-emerge a few frames        ║
# ║  later, the local minimum-cost match at the separation moment    ║
# ║  may pick the wrong post-merge mapping. After the fact, motion   ║
# ║  continuity over the next ~ second exposes the swap because a    ║
# ║  fly's extrapolated pre-merge motion no longer matches the       ║
# ║  centroid that was assigned its label.                           ║
# ║                                                                  ║
# ║  This module finds those "uncertainty events" (proximity         ║
# ║  contacts, NaN gaps), evaluates every permutation of post-event  ║
# ║  labels against the predicted-from-pre-event position, and       ║
# ║  commits a swap when a non-identity permutation reduces the      ║
# ║  motion residual by more than ``swap_cost_ratio``. Costs only    ║
# ║  use the position columns the cleaner already produces — no      ║
# ║  Kalman dependency. The motion model is a median-velocity        ║
# ║  predictor over the pre-event window; that is "lite" enough to   ║
# ║  ship today without Phase 4 yet a substantial improvement over  ║
# ║  pure local Hungarian.                                           ║
# ║                                                                  ║
# ║  References: Zhang/Li/Nevatia 2008 (min-cost flow MOT); Lenz et  ║
# ║  al. ICCV 2015 (sliding-window relaxation); Wang et al. 2019     ║
# ║  (muSSP fast min-cost flow). This is a pragmatic O(N!) audit at  ║
# ║  per-event granularity rather than a full graph solve, which is  ║
# ║  appropriate for fixed-N runs at N ≤ 8.                          ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Identity audit: re-tag track IDs by motion-residual permutation search."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SwapEvent:
    """One accepted swap: which permutation, where, by how much it helped."""

    frame_idx: int
    permutation: tuple[int, ...]
    cost_before: float
    cost_after: float


def audit_track_identities(
    traj: pd.DataFrame,
    *,
    fps: float,
    pix_per_mm: float,
    contact_threshold_mm: float = 5.0,
    window_s: float = 1.0,
    swap_cost_ratio: float = 0.7,
    coalesce_window_frames: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-tag ``track_id`` by re-optimising at proximity / gap events.

    Args:
        traj: Cleaned trajectory DataFrame (must contain ``frame_idx``,
            ``track_id``, ``cx_smooth_px``, ``cy_smooth_px``).
        fps: Source frame rate (used to size windows in seconds).
        pix_per_mm: Pixel-to-millimetre conversion (for the contact
            threshold).
        contact_threshold_mm: Two tracks closer than this are
            considered an "uncertainty event" worth auditing. Default
            5 mm ≈ a fly body length.
        window_s: Half-width of the pre-event and post-event windows
            in seconds. Default 1 s.
        swap_cost_ratio: A non-identity permutation is committed only
            if its motion-residual cost is below
            ``swap_cost_ratio × identity_cost``. Default 0.7
            (= 30% improvement). Lower → more conservative,
            higher → more aggressive.
        coalesce_window_frames: Adjacent event frames within this gap
            are merged into a single block. Defaults to ``round(fps *
            window_s)`` frames so a half-second gap is treated as one
            event.

    Returns:
        ``(relabelled_traj, swap_log)``. ``relabelled_traj`` is a
        copy of ``traj`` with ``track_id`` rewritten and a new
        ``original_track_id`` column for audit. ``swap_log`` is a
        DataFrame of the accepted swap events, one row per swap.
    """
    if pix_per_mm <= 0:
        raise ValueError(f"pix_per_mm must be positive, got {pix_per_mm}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if not (0.0 < swap_cost_ratio <= 1.0):
        raise ValueError(
            f"swap_cost_ratio must be in (0, 1], got {swap_cost_ratio}",
        )

    relabelled = traj.copy()
    relabelled["original_track_id"] = relabelled["track_id"]
    n_tracks = relabelled["track_id"].nunique()
    if n_tracks < 2:
        return relabelled, _empty_swap_log()

    track_ids = sorted(int(t) for t in relabelled["track_id"].unique())
    window_frames = max(3, int(round(fps * window_s)))
    if coalesce_window_frames is None:
        coalesce_window_frames = window_frames
    contact_threshold_px = contact_threshold_mm * pix_per_mm

    # Wide-form positions, columns ordered by sorted track_id. After this
    # column i always corresponds to whichever label is currently sitting
    # there; in-place column swaps below maintain that invariant.
    pivot_x = relabelled.pivot(
        index="frame_idx", columns="track_id", values="cx_smooth_px",
    ).reindex(columns=track_ids).sort_index()
    pivot_y = relabelled.pivot(
        index="frame_idx", columns="track_id", values="cy_smooth_px",
    ).reindex(columns=track_ids).sort_index()
    cx = pivot_x.to_numpy(dtype=float).copy()
    cy = pivot_y.to_numpy(dtype=float).copy()
    frames = pivot_x.index.to_numpy(dtype=np.int64)

    event_blocks = _find_event_blocks(
        cx, cy,
        contact_threshold_px=contact_threshold_px,
        coalesce_window_frames=coalesce_window_frames,
    )

    swaps: list[SwapEvent] = []
    identity = tuple(range(n_tracks))

    for block_start, block_end in event_blocks:
        pre_lo = max(0, block_start - window_frames)
        pre_hi = block_start                      # exclusive
        post_lo = block_end + 1
        post_hi = min(cx.shape[0], post_lo + window_frames)
        if pre_hi - pre_lo < 3 or post_hi - post_lo < 3:
            continue

        cost_id = _block_cost(cx, cy, pre_lo, pre_hi, post_lo, post_hi, identity)
        if cost_id <= 0.0:
            continue

        best_perm = identity
        best_cost = cost_id
        for perm in permutations(range(n_tracks)):
            if perm == identity:
                continue
            cost = _block_cost(cx, cy, pre_lo, pre_hi, post_lo, post_hi, perm)
            if cost < best_cost:
                best_cost = cost
                best_perm = perm

        if best_perm == identity:
            continue
        if best_cost > cost_id * swap_cost_ratio:
            continue

        # Apply the swap to the wide-form arrays so subsequent events
        # operate on the post-swap labelling.
        cols = list(best_perm)
        cx[post_lo:, :] = cx[post_lo:, cols]
        cy[post_lo:, :] = cy[post_lo:, cols]
        swaps.append(
            SwapEvent(
                frame_idx=int(frames[block_end]),
                permutation=best_perm,
                cost_before=float(cost_id),
                cost_after=float(best_cost),
            ),
        )

    # Apply swaps in chronological order to the DataFrame using the
    # current track_id at each step.
    for sw in swaps:
        mapping = {track_ids[i]: track_ids[sw.permutation[i]] for i in range(n_tracks)}
        mask = relabelled["frame_idx"] > sw.frame_idx
        relabelled.loc[mask, "track_id"] = (
            relabelled.loc[mask, "track_id"].map(mapping).astype(int)
        )

    swap_log = (
        pd.DataFrame(
            [{
                "frame_idx": sw.frame_idx,
                "permutation": list(sw.permutation),
                "cost_before": sw.cost_before,
                "cost_after": sw.cost_after,
                "improvement_ratio": (
                    1.0 - sw.cost_after / sw.cost_before if sw.cost_before > 0 else 0.0
                ),
            } for sw in swaps],
        )
        if swaps else _empty_swap_log()
    )
    return relabelled, swap_log


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────


def _empty_swap_log() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "frame_idx", "permutation",
            "cost_before", "cost_after", "improvement_ratio",
        ],
    )


def _find_event_blocks(
    cx: np.ndarray, cy: np.ndarray,
    *, contact_threshold_px: float, coalesce_window_frames: int,
) -> list[tuple[int, int]]:
    """Return ``[(start_idx, end_idx)]`` half-closed blocks of event frames."""
    n_frames, n_tracks = cx.shape
    if n_tracks < 2:
        return []

    # Pairwise distance per frame, excluding self.
    dx = cx[:, :, None] - cx[:, None, :]
    dy = cy[:, :, None] - cy[:, None, :]
    d = np.hypot(dx, dy)
    for i in range(n_tracks):
        d[:, i, i] = np.inf
    with np.errstate(invalid="ignore"):
        min_pair_d = np.nanmin(d.reshape(n_frames, -1), axis=1)
    contact = (min_pair_d < contact_threshold_px) & np.isfinite(min_pair_d)
    has_nan = np.any(np.isnan(cx) | np.isnan(cy), axis=1)
    event = contact | has_nan

    if not event.any():
        return []
    event_idx = np.where(event)[0]
    blocks: list[tuple[int, int]] = []
    cur_start = int(event_idx[0])
    cur_end = int(event_idx[0])
    for idx in event_idx[1:]:
        if int(idx) - cur_end <= coalesce_window_frames:
            cur_end = int(idx)
        else:
            blocks.append((cur_start, cur_end))
            cur_start = int(idx)
            cur_end = int(idx)
    blocks.append((cur_start, cur_end))
    return blocks


def _block_cost(
    cx: np.ndarray, cy: np.ndarray,
    pre_lo: int, pre_hi: int, post_lo: int, post_hi: int,
    perm: tuple[int, ...],
) -> float:
    """Mean predicted-vs-actual residual over the post window for ``perm``."""
    n_tracks = cx.shape[1]
    total = 0.0
    n_active = 0
    for i in range(n_tracks):
        pre_x = cx[pre_lo:pre_hi, i]
        pre_y = cy[pre_lo:pre_hi, i]
        valid_pre = ~(np.isnan(pre_x) | np.isnan(pre_y))
        if valid_pre.sum() < 2:
            continue
        # Median per-frame velocity (NaN-tolerant).
        vx = float(np.nanmedian(np.diff(pre_x)))
        vy = float(np.nanmedian(np.diff(pre_y)))
        if not (np.isfinite(vx) and np.isfinite(vy)):
            continue
        # Last observed pre-event position.
        last_idx_local = int(np.where(valid_pre)[0][-1])
        last_x = float(pre_x[last_idx_local])
        last_y = float(pre_y[last_idx_local])
        last_frame_offset = pre_lo + last_idx_local

        actual_x = cx[post_lo:post_hi, perm[i]]
        actual_y = cy[post_lo:post_hi, perm[i]]
        valid_post = ~(np.isnan(actual_x) | np.isnan(actual_y))
        if valid_post.sum() < 2:
            continue
        post_offsets = np.arange(post_lo, post_hi) - last_frame_offset
        pred_x = last_x + vx * post_offsets
        pred_y = last_y + vy * post_offsets
        res = np.hypot(
            actual_x[valid_post] - pred_x[valid_post],
            actual_y[valid_post] - pred_y[valid_post],
        )
        total += float(np.mean(res))
        n_active += 1

    if n_active == 0:
        return 0.0
    return total / n_active


__all__ = ["SwapEvent", "audit_track_identities"]

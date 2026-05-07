# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — posthoc.audit                                          ║
# ║  « sliding-window identity audit on Kalman-Mahalanobis cost »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Phase 6 of the post-hoc roadmap. The Hungarian tracker in       ║
# ║  pyLACE makes assignments one frame at a time; when two flies    ║
# ║  merge into a chain blob and re-emerge a few frames later, the   ║
# ║  local minimum-cost match at the separation moment may pick the  ║
# ║  wrong post-merge mapping. After the fact, motion continuity     ║
# ║  over the next ~ second exposes the swap because a fly's         ║
# ║  predicted pre-merge motion no longer matches the centroid that  ║
# ║  was assigned its label.                                         ║
# ║                                                                  ║
# ║  This module finds those "uncertainty events" (proximity         ║
# ║  contacts, NaN gaps), then for each event:                       ║
# ║    1. Replays a 4-state Kalman filter over the pre-event window  ║
# ║       per track (the same filter the tracker uses, fed only      ║
# ║       positions from the cleaned trajectory).                    ║
# ║    2. Predicts forward through the post-event window.            ║
# ║    3. Computes the squared Mahalanobis distance of the actual    ║
# ║       post-event position to the predicted state, under every    ║
# ║       permutation of post-event labels.                          ║
# ║    4. Commits the lowest-cost permutation iff it is non-identity ║
# ║       AND its mean Mahalanobis² is below ``swap_cost_ratio ×     ║
# ║       identity_cost`` (default 0.7 ⇒ require 30 % cost           ║
# ║       reduction).                                                ║
# ║                                                                  ║
# ║  Mahalanobis² scales with the inverse innovation covariance, so  ║
# ║  it discriminates much more sharply than the median-velocity    ║
# ║  Euclidean predictor used by the earlier "lite" version —        ║
# ║  particularly important for cool flies that barely move (the    ║
# ║  predictor sees ~ zero pre-event velocity and Euclidean cost     ║
# ║  alone struggles to tell two stationary tracks apart by          ║
# ║  motion).                                                        ║
# ║                                                                  ║
# ║  References: Bewley et al. SORT (ICIP 2016) for the 4-state CV   ║
# ║  filter; Zhang/Li/Nevatia 2008 (min-cost flow MOT); Lenz et al.  ║
# ║  ICCV 2015 (sliding-window relaxation). This is a pragmatic      ║
# ║  O(N!) audit at per-event granularity rather than a full graph   ║
# ║  solve, which is appropriate for fixed-N runs at N ≤ 8.          ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Identity audit: re-tag track IDs by Kalman-Mahalanobis permutation search."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
import pandas as pd

from pylace.tracking.constants import (
    DEFAULT_KALMAN_INITIAL_V_STD,
    DEFAULT_KALMAN_Q_POS,
    DEFAULT_KALMAN_Q_VEL,
    DEFAULT_KALMAN_R_POS,
)
from pylace.tracking.kalman import (
    initial_covariance,
    mahalanobis_sq,
    measurement_noise,
    predict,
    process_noise,
    transition_matrix,
    update,
)


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
    max_event_block_s: float | None = None,
    max_jump_mm: float | None = None,
    kalman_q_pos: float = DEFAULT_KALMAN_Q_POS,
    kalman_q_vel: float = DEFAULT_KALMAN_Q_VEL,
    kalman_r_pos: float = DEFAULT_KALMAN_R_POS,
    kalman_initial_v_std: float = DEFAULT_KALMAN_INITIAL_V_STD,
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
            if its mean post-event Mahalanobis² is below
            ``swap_cost_ratio × identity_cost``. Default 0.7
            (= 30% improvement). Lower → more conservative,
            higher → more aggressive.
        coalesce_window_frames: Adjacent event frames within this gap
            are merged into a single block. Defaults to ``round(fps *
            window_s)`` frames so a half-second gap is treated as one
            event.
        max_event_block_s: Reject (skip auditing) any event block
            longer than this many seconds. Long blocks bridge the
            Kalman filter through hundreds or thousands of
            predict-only steps; the covariance blows up so far that
            every permutation's Mahalanobis² collapses to numerical
            noise and the cost ratio becomes meaningless. Default
            ``5 × window_s`` (= 5 s at default settings) — long
            enough to audit a courtship-grade contact, short enough
            to refuse run-the-whole-video coalesced merges.
        max_jump_mm: Position-jump safety gate. A non-identity
            permutation is refused outright if any track's distance
            from its last valid pre-event position to the first valid
            post-event position (under the perm) exceeds this.
            Default ``2 × contact_threshold_mm`` (= 10 mm). Catches
            teleport-style swaps that the Kalman cost can rank
            favourably when the covariance is degenerate.
        kalman_q_pos, kalman_q_vel: Per-frame process-noise stds for
            the Kalman motion model used to score post-event
            assignments. Default to the same values as the tracker.
        kalman_r_pos: Per-axis measurement-noise std (px).
        kalman_initial_v_std: Initial velocity prior at filter
            initialisation (px/frame).

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
    if max_event_block_s is not None and max_event_block_s <= 0:
        raise ValueError(
            f"max_event_block_s must be positive, got {max_event_block_s}",
        )
    if max_jump_mm is not None and max_jump_mm <= 0:
        raise ValueError(
            f"max_jump_mm must be positive, got {max_jump_mm}",
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
    if max_event_block_s is None:
        max_event_block_frames = window_frames * 5
    else:
        max_event_block_frames = max(1, int(round(fps * max_event_block_s)))
    if max_jump_mm is None:
        max_jump_px = 2.0 * contact_threshold_px
    else:
        max_jump_px = float(max_jump_mm) * pix_per_mm

    # Kalman matrices, built once and shared across all event evaluations.
    F = transition_matrix(dt=1.0)
    Q = process_noise(kalman_q_pos, kalman_q_vel)
    R = measurement_noise(kalman_r_pos)
    init_cov = initial_covariance(kalman_r_pos, kalman_initial_v_std)
    kalman = (F, Q, R, init_cov)

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
        if block_end - block_start + 1 > max_event_block_frames:
            continue
        pre_lo = max(0, block_start - window_frames)
        pre_hi = block_start                      # exclusive
        post_lo = block_end + 1
        post_hi = min(cx.shape[0], post_lo + window_frames)
        if pre_hi - pre_lo < 3 or post_hi - post_lo < 3:
            continue

        cost_id = _block_cost(
            cx, cy, pre_lo, pre_hi, post_lo, post_hi, identity, kalman,
        )
        if cost_id <= 0.0:
            continue

        best_perm = identity
        best_cost = cost_id
        for perm in permutations(range(n_tracks)):
            if perm == identity:
                continue
            if _perm_max_jump_px(
                cx, cy, pre_lo, pre_hi, post_lo, post_hi, perm,
            ) > max_jump_px:
                continue
            cost = _block_cost(
                cx, cy, pre_lo, pre_hi, post_lo, post_hi, perm, kalman,
            )
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


def _default_kalman_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(F, Q, R, init_cov)`` at the package defaults — used by tests."""
    F = transition_matrix(dt=1.0)
    Q = process_noise(DEFAULT_KALMAN_Q_POS, DEFAULT_KALMAN_Q_VEL)
    R = measurement_noise(DEFAULT_KALMAN_R_POS)
    init_cov = initial_covariance(DEFAULT_KALMAN_R_POS, DEFAULT_KALMAN_INITIAL_V_STD)
    return F, Q, R, init_cov


def _perm_max_jump_px(
    cx: np.ndarray, cy: np.ndarray,
    pre_lo: int, pre_hi: int, post_lo: int, post_hi: int,
    perm: tuple[int, ...],
) -> float:
    """Worst-case per-track Euclidean jump from last-pre to first-post under ``perm``.

    Returns +inf if any track lacks a valid pre-event or post-event
    observation — without anchor points the jump is undefined and the
    perm should be refused.
    """
    n_tracks = cx.shape[1]
    worst = 0.0
    for i in range(n_tracks):
        last_pre = _last_valid_xy(cx[pre_lo:pre_hi, i], cy[pre_lo:pre_hi, i])
        first_post = _first_valid_xy(
            cx[post_lo:post_hi, perm[i]], cy[post_lo:post_hi, perm[i]],
        )
        if last_pre is None or first_post is None:
            return float("inf")
        d = float(np.hypot(last_pre[0] - first_post[0], last_pre[1] - first_post[1]))
        if d > worst:
            worst = d
    return worst


def _last_valid_xy(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    valid = ~(np.isnan(x) | np.isnan(y))
    if not valid.any():
        return None
    idx = int(np.where(valid)[0][-1])
    return float(x[idx]), float(y[idx])


def _first_valid_xy(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    valid = ~(np.isnan(x) | np.isnan(y))
    if not valid.any():
        return None
    idx = int(np.where(valid)[0][0])
    return float(x[idx]), float(y[idx])


def _block_cost(
    cx: np.ndarray, cy: np.ndarray,
    pre_lo: int, pre_hi: int, post_lo: int, post_hi: int,
    perm: tuple[int, ...],
    kalman: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> float:
    """Mean post-event Mahalanobis² of the perm assignment.

    For each track ``i``:
      1. Initialise a Kalman filter at the first valid pre-event
         observation.
      2. Replay predict + update through the pre-event window so the
         filter learns the per-track motion model.
      3. Predict forward to the first post-event frame.
      4. Score the post-event measurements for column ``perm[i]`` by
         their squared Mahalanobis distance from the predicted state;
         keep predicting (no update) so the cost reflects pure
         prediction quality, not measurement-driven adaptation.

    The returned cost is the per-track mean Mahalanobis², averaged
    across the active tracks.
    """
    if kalman is None:
        kalman = _default_kalman_matrices()
    F, Q, R, init_cov = kalman

    n_tracks = cx.shape[1]
    total = 0.0
    n_active = 0
    for i in range(n_tracks):
        pre_x = cx[pre_lo:pre_hi, i]
        pre_y = cy[pre_lo:pre_hi, i]
        valid_pre = ~(np.isnan(pre_x) | np.isnan(pre_y))
        if valid_pre.sum() < 2:
            continue

        first_idx = int(np.where(valid_pre)[0][0])
        last_idx = int(np.where(valid_pre)[0][-1])
        state = np.array(
            [float(pre_x[first_idx]), float(pre_y[first_idx]), 0.0, 0.0],
            dtype=np.float64,
        )
        cov = init_cov.copy()
        # Replay pre-event window: predict every frame, update on
        # observed frames.
        for j in range(first_idx + 1, pre_hi - pre_lo):
            state, cov = predict(state, cov, F, Q)
            if not (np.isnan(pre_x[j]) or np.isnan(pre_y[j])):
                z = np.array([pre_x[j], pre_y[j]], dtype=np.float64)
                state, cov, _ = update(state, cov, z, R)

        # Bridge to the first post-event frame.
        steps = post_lo - (pre_lo + last_idx)
        for _ in range(max(1, steps)):
            state, cov = predict(state, cov, F, Q)

        actual_x = cx[post_lo:post_hi, perm[i]]
        actual_y = cy[post_lo:post_hi, perm[i]]
        m_sq_sum = 0.0
        n_post_valid = 0
        for j in range(actual_x.size):
            if not (np.isnan(actual_x[j]) or np.isnan(actual_y[j])):
                z = np.array([actual_x[j], actual_y[j]], dtype=np.float64)
                m_sq = mahalanobis_sq(state, cov, z, R)
                m_sq_sum += m_sq
                n_post_valid += 1
            # Predict-only forward; no update so the cost reflects
            # pure prediction quality, not measurement-driven
            # adaptation that would dampen differences between perms.
            if j + 1 < actual_x.size:
                state, cov = predict(state, cov, F, Q)
        if n_post_valid == 0:
            continue
        total += m_sq_sum / n_post_valid
        n_active += 1

    if n_active == 0:
        return 0.0
    return total / n_active


__all__ = ["SwapEvent", "audit_track_identities"]


# Two safety gates were added on 2026-05-07 after a degenerate-case
# false swap on arena_04 (frame 44061): a coalesced event block
# spanning ~42 000 frames bridged the Kalman filter through enough
# predict-only steps that the covariance grew to ~1e10 px², every
# permutation's Mahalanobis² collapsed to ~1e-7, and the cost-ratio
# rule committed a 390-px teleport between two non-contacting flies.
#
# Gate 1 — max_event_block_s: skip any event block longer than the
#   configured duration (default 5 × window_s = 5 s). Real fly
#   contacts last under a few seconds; multi-minute coalesced blocks
#   are degenerate by definition and the audit cannot resolve them.
#
# Gate 2 — max_jump_mm: refuse any non-identity permutation whose
#   worst-case track jump from last-pre to first-post position
#   exceeds the threshold (default 2 × contact_threshold_mm). The
#   gate is a belt-and-braces guard independent of the Kalman cost
#   so a numerically-degenerate cost ratio cannot commit an obvious
#   teleport.

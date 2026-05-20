# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — review.candidates                                      ║
# ║  « cheap, audit-independent detector of swap-candidate events »  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  The audit's permutation logic already produces a swap log, but  ║
# ║  it only fires on events where the cost ratio justifies a        ║
# ║  commit. The reviewer also wants to see events that the audit    ║
# ║  was unsure about, or that the audit gates filtered out          ║
# ║  (long-contact mounts, teleport-jumps refused as biology). This  ║
# ║  module scans a cleaned trajectory and emits one candidate per   ║
# ║  block — independent of any audit run.                           ║
# ║                                                                  ║
# ║  Two cheap detectors, mirrored on the audit's internals:         ║
# ║    1. contact — pairwise distance < contact_threshold_mm         ║
# ║    2. jump    — single-frame Euclidean step > jump_threshold_mm  ║
# ║  Both flavours collapse adjacent event frames into one block.    ║
# ║  Source-of-trigger is recorded per candidate so the GUI can      ║
# ║  prioritise (e.g. contact + jump together = high suspicion).     ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Swap-candidate detector and on-disk format for the swap-review GUI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pylace.review.verdicts import make_event_id

CANDIDATES_SUFFIX = ".pylace_candidates.csv"

# What the per-row 'source' set may contain. Recorded as a comma-joined
# string of these tokens.
SOURCE_CONTACT = "contact"
SOURCE_JUMP = "jump"

_COLUMNS: tuple[str, ...] = (
    "event_id", "frame_start", "frame_end",
    "animal_a", "animal_b",
    "source",
    "min_distance_px", "max_jump_px", "n_frames",
)


@dataclass(frozen=True)
class Candidate:
    """One swap-candidate event spanning a contiguous block of frames."""

    event_id: str
    frame_start: int
    frame_end: int
    animal_a: int
    animal_b: int
    source: str
    min_distance_px: float
    max_jump_px: float
    n_frames: int


def detect_candidates(
    traj: pd.DataFrame,
    *,
    fps: float,
    pix_per_mm: float,
    contact_threshold_mm: float = 5.0,
    jump_threshold_mm_per_frame: float = 5.0,
    coalesce_window_frames: int | None = None,
    window_s: float = 1.0,
) -> list[Candidate]:
    """Scan a cleaned trajectory and return one candidate per event block.

    Args:
        traj: DataFrame with ``frame_idx``, ``track_id``,
            ``cx_smooth_px``, ``cy_smooth_px``.
        fps: Source frame rate (used for the default coalesce window).
        pix_per_mm: Pixel-to-millimetre conversion.
        contact_threshold_mm: Two tracks closer than this trigger a
            ``contact`` event. Default 5 mm (≈ a fly body length).
        jump_threshold_mm_per_frame: A single-frame Euclidean step in
            either track larger than this triggers a ``jump`` event.
            Default 5 mm/frame (= ~150 mm/s at 30 fps); flies typically
            sit well below this except during very fast saccades.
        coalesce_window_frames: Adjacent event frames within this gap
            merge into one block. Defaults to ``round(fps * window_s)``,
            matching the audit's coalescing.
        window_s: Used only for the default coalesce window.

    Returns:
        Sorted list of :class:`Candidate`. Empty list if no events.
    """
    if pix_per_mm <= 0:
        raise ValueError(f"pix_per_mm must be positive, got {pix_per_mm}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if contact_threshold_mm < 0:
        raise ValueError(
            f"contact_threshold_mm must be >= 0, got {contact_threshold_mm}",
        )
    if jump_threshold_mm_per_frame <= 0:
        raise ValueError(
            "jump_threshold_mm_per_frame must be positive, "
            f"got {jump_threshold_mm_per_frame}",
        )

    n_tracks = traj["track_id"].nunique()
    if n_tracks < 2:
        return []
    window_frames = max(3, int(round(fps * window_s)))
    if coalesce_window_frames is None:
        coalesce_window_frames = window_frames
    contact_threshold_px = contact_threshold_mm * pix_per_mm
    jump_threshold_px = jump_threshold_mm_per_frame * pix_per_mm

    track_ids = sorted(int(t) for t in traj["track_id"].unique())
    pivot_x = traj.pivot(
        index="frame_idx", columns="track_id", values="cx_smooth_px",
    ).reindex(columns=track_ids).sort_index()
    pivot_y = traj.pivot(
        index="frame_idx", columns="track_id", values="cy_smooth_px",
    ).reindex(columns=track_ids).sort_index()
    cx = pivot_x.to_numpy(dtype=float)
    cy = pivot_y.to_numpy(dtype=float)
    frames = pivot_x.index.to_numpy(dtype=np.int64)

    n_frames, n_t = cx.shape

    # Pairwise distance and per-frame jump magnitudes.
    dx = cx[:, :, None] - cx[:, None, :]
    dy = cy[:, :, None] - cy[:, None, :]
    pair_d = np.hypot(dx, dy)
    for i in range(n_t):
        pair_d[:, i, i] = np.inf

    step_x = np.diff(cx, axis=0, prepend=cx[:1])
    step_y = np.diff(cy, axis=0, prepend=cy[:1])
    step_mag = np.hypot(step_x, step_y)

    has_nan = np.isnan(cx) | np.isnan(cy)
    any_nan = has_nan.any(axis=1)

    contact_mask = (pair_d < contact_threshold_px).any(axis=(1, 2))
    contact_mask &= np.isfinite(pair_d.reshape(n_frames, -1)).any(axis=1)
    jump_mask = (
        (step_mag > jump_threshold_px) & ~has_nan
    ).any(axis=1)
    event_mask = contact_mask | jump_mask | any_nan
    if not event_mask.any():
        return []

    event_idx = np.where(event_mask)[0]
    spans: list[tuple[int, int]] = []
    cur_start = int(event_idx[0])
    cur_end = int(event_idx[0])
    for idx in event_idx[1:]:
        if int(idx) - cur_end <= coalesce_window_frames:
            cur_end = int(idx)
        else:
            spans.append((cur_start, cur_end))
            cur_start = int(idx)
            cur_end = int(idx)
    spans.append((cur_start, cur_end))

    candidates: list[Candidate] = []
    for s, e in spans:
        block_pair_d = pair_d[s:e + 1]
        with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "Mean of empty slice")
            mean_pair = np.nanmean(block_pair_d, axis=0)
            min_pair = np.nanmin(block_pair_d, axis=0)
        for i in range(n_t):
            mean_pair[i, i] = np.inf
            min_pair[i, i] = np.inf
        mean_pair = np.where(np.isnan(mean_pair), np.inf, mean_pair)
        min_pair = np.where(np.isnan(min_pair), np.inf, min_pair)
        flat = int(np.argmin(mean_pair))
        col_a, col_b = divmod(flat, n_t)
        if col_a > col_b:
            col_a, col_b = col_b, col_a
        tid_a = track_ids[col_a]
        tid_b = track_ids[col_b]

        sources: list[str] = []
        if contact_mask[s:e + 1].any():
            sources.append(SOURCE_CONTACT)
        if jump_mask[s:e + 1].any():
            sources.append(SOURCE_JUMP)
        if not sources:
            # Pure-NaN span: report it as contact so the GUI shows it.
            sources.append(SOURCE_CONTACT)
        block_min_d = float(min_pair[col_a, col_b])
        if not np.isfinite(block_min_d):
            block_min_d = float("nan")
        block_max_jump = float(np.nanmax(step_mag[s:e + 1, [col_a, col_b]]))
        if not np.isfinite(block_max_jump):
            block_max_jump = float("nan")
        block_start_frame = int(frames[s])
        block_end_frame = int(frames[e])
        candidates.append(Candidate(
            event_id=make_event_id(block_start_frame, tid_a, tid_b),
            frame_start=block_start_frame,
            frame_end=block_end_frame,
            animal_a=tid_a,
            animal_b=tid_b,
            source=",".join(sources),
            min_distance_px=block_min_d,
            max_jump_px=block_max_jump,
            n_frames=int(e - s + 1),
        ))
    return candidates


def default_candidates_path(audited_csv: Path) -> Path:
    """Conventional candidates CSV next to the audited / trajectory CSV."""
    from pylace.posthoc.io import trajectory_stem
    audited_csv = Path(audited_csv)
    return audited_csv.with_name(trajectory_stem(audited_csv) + CANDIDATES_SUFFIX)


def write_candidates(
    candidates: Iterable[Candidate], path: Path,
) -> None:
    """Atomically replace the candidates CSV at ``path``."""
    candidates = list(candidates)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "event_id": c.event_id,
                "frame_start": int(c.frame_start),
                "frame_end": int(c.frame_end),
                "animal_a": int(c.animal_a),
                "animal_b": int(c.animal_b),
                "source": c.source,
                "min_distance_px": float(c.min_distance_px),
                "max_jump_px": float(c.max_jump_px),
                "n_frames": int(c.n_frames),
            }
            for c in candidates
        ],
        columns=list(_COLUMNS),
    )
    if not df.empty:
        df = df.sort_values(["frame_start", "event_id"]).reset_index(drop=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, float_format="%.4f")
    os.replace(tmp, path)


def read_candidates(path: Path) -> list[Candidate]:
    """Load a candidates CSV. Missing file → empty list."""
    path = Path(path)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    missing = [c for c in _COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Candidates CSV {path} missing columns: {missing}")
    out: list[Candidate] = []
    for _, row in df.iterrows():
        out.append(Candidate(
            event_id=str(row["event_id"]),
            frame_start=int(row["frame_start"]),
            frame_end=int(row["frame_end"]),
            animal_a=int(row["animal_a"]),
            animal_b=int(row["animal_b"]),
            source=str(row["source"]),
            min_distance_px=float(row["min_distance_px"]),
            max_jump_px=float(row["max_jump_px"]),
            n_frames=int(row["n_frames"]),
        ))
    return out


__all__ = [
    "CANDIDATES_SUFFIX",
    "Candidate",
    "SOURCE_CONTACT",
    "SOURCE_JUMP",
    "default_candidates_path",
    "detect_candidates",
    "read_candidates",
    "write_candidates",
]

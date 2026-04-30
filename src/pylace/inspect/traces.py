# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — inspect.traces                                         ║
# ║  « load detections CSV; render trails and trajectories »         ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Pure-numpy / cv2 helpers, no Qt. The main window calls these    ║
# ║  to draw the live trail and the static overview; tests can       ║
# ║  exercise them without a Qt event loop.                          ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Read detection traces and render them onto BGR frames."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class TrackTrajectory:
    """All positions a track was detected at, indexed by absolute frame."""

    track_id: int
    frame_indices: np.ndarray  # int (N,)
    cx_px: np.ndarray  # float (N,)
    cy_px: np.ndarray  # float (N,)


def read_traces(csv_path: Path) -> list[TrackTrajectory]:
    """Group ``pylace-detect`` CSV rows by ``track_id``, sorted by frame.

    Args:
        csv_path: Path to the detections CSV.

    Returns:
        One ``TrackTrajectory`` per distinct ``track_id``, ordered by id.
    """
    rows_by_track: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_track[int(row["track_id"])].append(
                (int(row["frame_idx"]), float(row["cx_px"]), float(row["cy_px"])),
            )
    out: list[TrackTrajectory] = []
    for tid in sorted(rows_by_track.keys()):
        rows = sorted(rows_by_track[tid], key=lambda r: r[0])
        out.append(
            TrackTrajectory(
                track_id=tid,
                frame_indices=np.array([r[0] for r in rows], dtype=np.int64),
                cx_px=np.array([r[1] for r in rows], dtype=np.float64),
                cy_px=np.array([r[2] for r in rows], dtype=np.float64),
            )
        )
    return out


def render_full_trajectories(
    bgr: np.ndarray,
    trajectories: list[TrackTrajectory],
    colours: list[tuple[int, int, int]],
    *,
    line_thickness: int = 1,
) -> None:
    """Draw every track's full trajectory as a polyline on ``bgr`` in place.

    Used for the overview panel: a single static draw over the chosen
    background image.
    """
    for traj, colour in zip(trajectories, colours, strict=False):
        if traj.cx_px.size < 2:
            continue
        pts = np.column_stack(
            (traj.cx_px.astype(np.int32), traj.cy_px.astype(np.int32)),
        ).reshape(-1, 1, 2)
        cv2.polylines(bgr, [pts], isClosed=False, color=colour, thickness=line_thickness)


def render_trail(
    bgr: np.ndarray,
    trajectory: TrackTrajectory,
    current_frame: int,
    trail_frames: int,
    colour: tuple[int, int, int],
    *,
    line_thickness: int = 2,
    min_alpha: float = 0.1,
) -> None:
    """Draw an ``trail_frames``-long fading trail leading up to ``current_frame``.

    Older segments are rendered in a darker shade of ``colour`` (intensity
    decay rather than true alpha — looks correct on a video frame and
    avoids per-segment image blends).
    """
    if trail_frames <= 0 or trajectory.cx_px.size < 2:
        return
    lo = current_frame - trail_frames
    mask = (trajectory.frame_indices >= lo) & (trajectory.frame_indices <= current_frame)
    sel_frames = trajectory.frame_indices[mask]
    sel_cx = trajectory.cx_px[mask]
    sel_cy = trajectory.cy_px[mask]
    if sel_frames.size < 2:
        return
    for i in range(sel_frames.size - 1):
        age = current_frame - int(sel_frames[i])
        alpha = max(min_alpha, 1.0 - age / trail_frames)
        dimmed = tuple(int(round(c * alpha)) for c in colour)
        cv2.line(
            bgr,
            (int(sel_cx[i]), int(sel_cy[i])),
            (int(sel_cx[i + 1]), int(sel_cy[i + 1])),
            dimmed, line_thickness,
        )


def render_current_markers(
    bgr: np.ndarray,
    trajectories: list[TrackTrajectory],
    colours: list[tuple[int, int, int]],
    current_frame: int,
    *,
    radius_px: int = 6,
    label: bool = True,
) -> None:
    """Mark each track's position at ``current_frame`` (if present) as a filled circle."""
    for traj, colour in zip(trajectories, colours, strict=False):
        idx = np.where(traj.frame_indices == current_frame)[0]
        if idx.size == 0:
            continue
        i = int(idx[0])
        x = int(traj.cx_px[i])
        y = int(traj.cy_px[i])
        cv2.circle(bgr, (x, y), radius_px, colour, thickness=-1)
        cv2.circle(bgr, (x, y), radius_px, (0, 0, 0), thickness=1)
        if label:
            cv2.putText(
                bgr, str(traj.track_id),
                (x + radius_px + 2, y - radius_px),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
            )


__all__ = [
    "TrackTrajectory",
    "read_traces",
    "render_current_markers",
    "render_full_trajectories",
    "render_trail",
]

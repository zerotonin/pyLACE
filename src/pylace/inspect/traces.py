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

from pylace.annotator.geometry import Arena, Circle
from pylace.roi.geometry import ROISet

ARENA_OUTLINE_COLOR = (0, 220, 220)        # cyan
ROI_ADD_OUTLINE_COLOR = (0, 220, 0)        # green
ROI_SUB_OUTLINE_COLOR = (0, 0, 220)        # red
FREEHAND_OUTLINE_COLOR = (200, 220, 0)     # teal


@dataclass
class TrackTrajectory:
    """All positions a track was detected at, indexed by absolute frame.

    ``heading_deg`` is optional — populated when the source CSV has a
    ``heading_deg`` column (i.e. cleaned / audited trajectories) and
    left as ``None`` for raw detection CSVs.
    """

    track_id: int
    frame_indices: np.ndarray  # int (N,)
    cx_px: np.ndarray  # float (N,)
    cy_px: np.ndarray  # float (N,)
    heading_deg: np.ndarray | None = None  # float (N,) or None


def read_traces(csv_path: Path) -> list[TrackTrajectory]:
    """Group ``pylace-detect`` CSV rows by ``track_id``, sorted by frame.

    Args:
        csv_path: Path to the detections CSV.

    Returns:
        One ``TrackTrajectory`` per distinct ``track_id``, ordered by id.
    """
    rows_by_track: dict[int, list[tuple[int, float, float, float]]] = defaultdict(list)
    has_heading = False
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is not None:
            has_heading = "heading_deg" in reader.fieldnames
        for row in reader:
            heading = float(row["heading_deg"]) if has_heading else float("nan")
            rows_by_track[int(row["track_id"])].append(
                (int(row["frame_idx"]), float(row["cx_px"]),
                 float(row["cy_px"]), heading),
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
                heading_deg=(
                    np.array([r[3] for r in rows], dtype=np.float64)
                    if has_heading else None
                ),
            )
        )
    return out


def render_full_trajectories(
    bgr: np.ndarray,
    trajectories: list[TrackTrajectory],
    colours: list[tuple[int, int, int]],
    *,
    line_thickness: int = 1,
    max_gap_frames: int = 25,
    max_jump_px: float = 50.0,
) -> None:
    """Draw every track's full trajectory as a polyline on ``bgr`` in place.

    The polyline is broken whenever consecutive samples are more than
    ``max_gap_frames`` apart in time OR more than ``max_jump_px`` apart
    in space. NaN positions (introduced by the post-hoc cleaner for
    unbridged big gaps) are dropped before the segment splitter runs,
    so a NaN never reaches ``astype(np.int32)``. Defaults: 25 frames
    (≈ 1 s at typical fps) and 50 px (well above any plausible
    per-frame fly motion).
    """
    for traj, colour in zip(trajectories, colours, strict=False):
        if traj.cx_px.size < 2:
            continue
        valid = ~(np.isnan(traj.cx_px) | np.isnan(traj.cy_px))
        if valid.sum() < 2:
            continue
        v_cx = traj.cx_px[valid]
        v_cy = traj.cy_px[valid]
        v_fr = traj.frame_indices[valid]
        for s, e in _segment_indices(
            v_fr, v_cx, v_cy,
            max_gap_frames=max_gap_frames, max_jump_px=max_jump_px,
        ):
            if e - s < 2:
                continue
            pts = np.column_stack(
                (
                    v_cx[s:e].astype(np.int32),
                    v_cy[s:e].astype(np.int32),
                ),
            ).reshape(-1, 1, 2)
            cv2.polylines(
                bgr, [pts], isClosed=False, color=colour,
                thickness=line_thickness,
            )


def _segment_indices(
    frame_indices: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    *,
    max_gap_frames: int,
    max_jump_px: float,
) -> list[tuple[int, int]]:
    """Return ``(start, end)`` half-open ranges split at frame OR position breaks.

    Caller must pass NaN-free ``cx``/``cy`` arrays — the splitter is
    purely a frame-gap / position-jump detector.
    """
    if frame_indices.size == 0:
        return []
    frame_gap = np.diff(frame_indices) > max_gap_frames
    pos_jump = np.hypot(np.diff(cx), np.diff(cy)) > max_jump_px
    breaks = np.where(frame_gap | pos_jump)[0] + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [frame_indices.size]))
    return list(zip(starts.tolist(), ends.tolist()))


def render_trail(
    bgr: np.ndarray,
    trajectory: TrackTrajectory,
    current_frame: int,
    trail_frames: int,
    colour: tuple[int, int, int],
    *,
    line_thickness: int = 2,
    min_alpha: float = 0.1,
    max_gap_frames: int = 5,
) -> None:
    """Draw an ``trail_frames``-long fading trail leading up to ``current_frame``.

    Segments whose endpoints are more than ``max_gap_frames`` apart are
    skipped so the trail does not bridge an occlusion gap with a straight
    line. Older segments are rendered in a darker shade of ``colour``
    (intensity decay rather than true alpha — looks correct on a video
    frame and avoids per-segment image blends).
    """
    if trail_frames <= 0 or trajectory.cx_px.size < 2:
        return
    lo = current_frame - trail_frames
    mask = (trajectory.frame_indices >= lo) & (trajectory.frame_indices <= current_frame)
    sel_frames = trajectory.frame_indices[mask]
    sel_cx = trajectory.cx_px[mask]
    sel_cy = trajectory.cy_px[mask]
    valid = ~(np.isnan(sel_cx) | np.isnan(sel_cy))
    sel_frames = sel_frames[valid]
    sel_cx = sel_cx[valid]
    sel_cy = sel_cy[valid]
    if sel_frames.size < 2:
        return
    for i in range(sel_frames.size - 1):
        if int(sel_frames[i + 1]) - int(sel_frames[i]) > max_gap_frames:
            continue
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
    show_orientation: bool = True,
    orientation_length_px: int = 18,
) -> None:
    """Mark each track's position at ``current_frame`` as a filled circle.

    When ``show_orientation`` is True and the trajectory carries a
    ``heading_deg`` array (cleaned / audited CSVs), a short arrow is
    drawn from the centroid in the heading direction so the user can
    see body axis at a glance.
    """
    for traj, colour in zip(trajectories, colours, strict=False):
        idx = np.where(traj.frame_indices == current_frame)[0]
        if idx.size == 0:
            continue
        i = int(idx[0])
        if np.isnan(traj.cx_px[i]) or np.isnan(traj.cy_px[i]):
            continue
        x = int(traj.cx_px[i])
        y = int(traj.cy_px[i])
        if (
            show_orientation
            and traj.heading_deg is not None
            and i < traj.heading_deg.size
            and not np.isnan(traj.heading_deg[i])
        ):
            theta = np.deg2rad(float(traj.heading_deg[i]))
            tip_x = int(round(x + orientation_length_px * np.cos(theta)))
            tip_y = int(round(y + orientation_length_px * np.sin(theta)))
            cv2.arrowedLine(
                bgr, (x, y), (tip_x, tip_y),
                colour, thickness=2, tipLength=0.4,
            )
        cv2.circle(bgr, (x, y), radius_px, colour, thickness=-1)
        cv2.circle(bgr, (x, y), radius_px, (0, 0, 0), thickness=1)
        if label:
            cv2.putText(
                bgr, str(traj.track_id),
                (x + radius_px + 2, y - radius_px),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
            )


def render_arena_outline(
    bgr: np.ndarray,
    arena: Arena,
    *,
    color: tuple[int, int, int] = ARENA_OUTLINE_COLOR,
    thickness: int = 1,
) -> None:
    """Draw the arena boundary onto ``bgr`` in place."""
    if isinstance(arena, Circle):
        cv2.circle(
            bgr,
            (int(round(arena.cx)), int(round(arena.cy))),
            int(round(arena.r)),
            color, thickness,
        )
        return
    poly = np.array(arena.vertices, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(bgr, [poly], isClosed=True, color=color, thickness=thickness)


def render_roi_outlines(
    bgr: np.ndarray,
    roi_set: ROISet,
    *,
    add_color: tuple[int, int, int] = ROI_ADD_OUTLINE_COLOR,
    subtract_color: tuple[int, int, int] = ROI_SUB_OUTLINE_COLOR,
    freehand_color: tuple[int, int, int] = FREEHAND_OUTLINE_COLOR,
    thickness: int = 1,
) -> None:
    """Draw every ROI outline onto ``bgr`` in place; colour by add/subtract."""
    for roi in roi_set.rois:
        col = add_color if roi.operation == "add" else subtract_color
        render_arena_outline(bgr, roi.shape, color=col, thickness=thickness)
    if roi_set.has_freehand_mask():
        mask = roi_set.freehand_mask.astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(bgr, contours, -1, freehand_color, thickness)


__all__ = [
    "ARENA_OUTLINE_COLOR",
    "FREEHAND_OUTLINE_COLOR",
    "ROI_ADD_OUTLINE_COLOR",
    "ROI_SUB_OUTLINE_COLOR",
    "TrackTrajectory",
    "read_traces",
    "render_arena_outline",
    "render_current_markers",
    "render_full_trajectories",
    "render_roi_outlines",
    "render_trail",
]

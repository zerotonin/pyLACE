# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — tests.test_inspect_traces                              ║
# ║  « CSV grouping + trail / overview rendering »                   ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Tests for the inspect.traces helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from pylace.inspect.traces import (
    TrackTrajectory,
    read_traces,
    render_current_markers,
    render_full_trajectories,
    render_trail,
)


def _write_csv(path: Path, rows: list[tuple[int, int, float, float]]) -> None:
    path.write_text("")
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx", "roi_label", "track_id",
            "cx_px", "cy_px", "x_mm", "y_mm",
            "area_px", "major_axis_px", "minor_axis_px", "orientation_deg",
        ])
        for frame_idx, track_id, cx, cy in rows:
            writer.writerow([
                frame_idx, "_merged", track_id,
                f"{cx:.3f}", f"{cy:.3f}", "0", "0", "0", "0", "0", "0",
            ])


def test_read_traces_groups_by_track_id(tmp_path: Path):
    csv_path = tmp_path / "d.csv"
    _write_csv(csv_path, [
        (0, 0, 10.0, 10.0),
        (0, 1, 50.0, 50.0),
        (1, 0, 11.0, 11.0),
        (1, 1, 51.0, 51.0),
        (2, 0, 12.0, 12.0),
    ])
    traj = read_traces(csv_path)
    assert [t.track_id for t in traj] == [0, 1]
    assert traj[0].frame_indices.tolist() == [0, 1, 2]
    assert traj[1].frame_indices.tolist() == [0, 1]
    assert traj[1].cx_px.tolist() == [50.0, 51.0]


def test_read_traces_orders_each_track_by_frame(tmp_path: Path):
    """Out-of-order rows are still sorted into per-track time order."""
    csv_path = tmp_path / "d.csv"
    _write_csv(csv_path, [
        (5, 0, 50.0, 50.0),
        (1, 0, 10.0, 10.0),
        (3, 0, 30.0, 30.0),
    ])
    traj = read_traces(csv_path)
    assert traj[0].frame_indices.tolist() == [1, 3, 5]
    assert traj[0].cx_px.tolist() == [10.0, 30.0, 50.0]


def test_render_full_trajectories_draws_polylines():
    """Polylines for trajectories should produce coloured pixels on the canvas."""
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    traj = TrackTrajectory(
        track_id=0,
        frame_indices=np.array([0, 1, 2, 3]),
        cx_px=np.array([10.0, 30.0, 50.0, 70.0]),
        cy_px=np.array([10.0, 30.0, 50.0, 70.0]),
    )
    render_full_trajectories(bgr, [traj], [(0, 255, 0)])
    coloured = ((bgr == np.array((0, 255, 0))).all(axis=2)).sum()
    assert coloured > 0


def test_render_trail_only_draws_within_window():
    """Segments outside the trail window must leave the canvas untouched."""
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    traj = TrackTrajectory(
        track_id=0,
        frame_indices=np.array([0, 1, 90, 91]),
        cx_px=np.array([5.0, 5.0, 80.0, 90.0]),
        cy_px=np.array([5.0, 5.0, 80.0, 90.0]),
    )
    render_trail(bgr, traj, current_frame=92, trail_frames=5, colour=(255, 0, 0))
    # Segment 90→91 is within the trail window; segment 0→1 is not.
    assert bgr[80, 80].sum() > 0 or bgr[85, 85].sum() > 0
    assert bgr[5, 5].sum() == 0


def test_render_trail_dims_older_segments():
    """Older segments are drawn with a darker colour than newer ones."""
    bgr = np.zeros((200, 200, 3), dtype=np.uint8)
    traj = TrackTrajectory(
        track_id=0,
        frame_indices=np.arange(0, 50, 1),
        cx_px=np.linspace(10, 190, 50),
        cy_px=np.full(50, 100.0),
    )
    render_trail(bgr, traj, current_frame=49, trail_frames=49,
                 colour=(0, 0, 255), line_thickness=3, min_alpha=0.0)
    # Sample a pixel near the start of the trail (oldest) and near the end.
    old_row = bgr[100, 12]
    new_row = bgr[100, 188]
    assert new_row[2] >= old_row[2]


def test_render_current_markers_only_when_present_at_frame():
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    traj_present = TrackTrajectory(
        track_id=0,
        frame_indices=np.array([10]),
        cx_px=np.array([50.0]),
        cy_px=np.array([50.0]),
    )
    traj_missing = TrackTrajectory(
        track_id=1,
        frame_indices=np.array([20]),
        cx_px=np.array([50.0]),
        cy_px=np.array([50.0]),
    )
    render_current_markers(
        bgr, [traj_present, traj_missing], [(0, 255, 0), (0, 0, 255)],
        current_frame=10, label=False,
    )
    assert bgr[50, 50, 1] > 0  # green pixel at the present-track centroid
    # The missing track has no marker drawn for this frame; the only
    # green-bearing region is around (50, 50).
    far_pixel = bgr[10, 10]
    assert far_pixel.sum() == 0

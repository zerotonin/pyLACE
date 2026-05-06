"""Smoke test for the posthoc explorer window — instantiates offscreen."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Force offscreen Qt before importing PyQt anything.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def fake_dataset(tmp_path: Path):
    """Build a minimal video stub + sidecar + trajectory CSV in tmp_path."""
    from pylace.annotator.geometry import Calibration, Circle, WorldFrame
    from pylace.annotator.sidecar import Sidecar, VideoMeta, write_sidecar

    video = tmp_path / "fake.mp4"
    video.write_bytes(b"not a real video")
    sidecar = Sidecar(
        video=VideoMeta(
            path=str(video), sha256="0" * 64,
            frame_size=(200, 200), fps=25.0,
        ),
        arena=Circle(cx=100.0, cy=100.0, r=80.0),
        world_frame=WorldFrame(origin_pixel=(100.0, 100.0)),
        calibration=Calibration(
            reference_kind="diameter",
            physical_mm=20.0, pixel_distance=160.0,
        ),
    )
    sidecar_path = tmp_path / "fake.mp4.pylace_arena.json"
    write_sidecar(sidecar, sidecar_path)

    csv_path = tmp_path / "fake.mp4.pylace_trajectory.csv"
    with csv_path.open("w") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_idx", "roi_label", "track_id",
            "cx_px", "cy_px", "x_mm", "y_mm",
            "area_px", "perimeter_px", "solidity",
            "major_axis_px", "minor_axis_px", "orientation_deg",
            "cx_smooth_px", "cy_smooth_px",
            "vx_mm_s", "vy_mm_s", "speed_mm_s",
            "heading_deg", "yaw_deg", "yaw_rate_deg_s",
            "interpolated", "outlier_rejected", "heading_unresolved",
        ])
        for fr in range(0, 50):
            for tid in (0, 1):
                cx = 100.0 + 30.0 * np.cos(fr * 0.1 + tid * 1.5)
                cy = 100.0 + 30.0 * np.sin(fr * 0.1 + tid * 1.5)
                w.writerow([
                    fr, "_merged", tid,
                    cx, cy, 0.0, 0.0,
                    2500, 180, 0.9,
                    58, 22, 0.0,
                    cx, cy,
                    1.0, 1.0, 1.4,
                    0.0, 0.0, 0.0,
                    False, False, False,
                ])
    return video, csv_path, sidecar_path


def test_explorer_window_constructs_offscreen(fake_dataset):
    video, csv_path, sidecar_path = fake_dataset
    from PyQt6 import QtWidgets

    from pylace.annotator.sidecar import read_sidecar
    from pylace.posthoc.explorer import ExplorerWindow

    with patch("pylace.posthoc.explorer.cv2.VideoCapture") as mc:
        cap = mc.return_value
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda p: {3: 200, 4: 200, 5: 25.0, 7: 50}.get(p, 0)
        cap.read.return_value = (True, np.zeros((200, 200, 3), dtype=np.uint8))
        sc = read_sidecar(sidecar_path)
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        w = ExplorerWindow(
            video=video, trajectory_csv=csv_path,
            sidecar=sc, roi_set=None,
        )
        # Two tracks, one per id.
        assert sorted(w._track_ids) == [0, 1]
        # Three matplotlib axes, three vertical cursor lines.
        assert len(w._cursor_lines) == 3
        # Navigation strip wired and aware of the video length.
        assert w._nav.current_frame() in range(50)
        # The synthetic centroid for track 0 at frame 5 is at
        # (100 + 30*cos(0.5), 100 + 30*sin(0.5)) ≈ (126.3, 114.4).
        w._on_pane_clicked(126.3, 114.4)
        assert w._current_frame == 5
        w._set_current_frame(20, source="test")
        assert w._current_frame == 20
        assert w._nav.current_frame() == 20

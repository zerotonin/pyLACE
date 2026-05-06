"""CSV round-trip for the cleaner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pylace.posthoc.io import (
    DETECTION_COLUMNS,
    default_trajectory_path,
    read_detections,
    write_trajectory,
)


def _write_minimal_detections(path: Path, n: int = 5) -> None:
    df = pd.DataFrame({
        "frame_idx": np.arange(n),
        "roi_label": ["_merged"] * n,
        "track_id": [0] * n,
        "cx_px": np.arange(n, dtype=float),
        "cy_px": np.arange(n, dtype=float) * 2.0,
        "x_mm": np.arange(n, dtype=float) * 0.1,
        "y_mm": np.arange(n, dtype=float) * 0.2,
        "area_px": np.full(n, 2500.0),
        "perimeter_px": np.full(n, 180.0),
        "solidity": np.full(n, 0.9),
        "major_axis_px": np.full(n, 58.0),
        "minor_axis_px": np.full(n, 22.0),
        "orientation_deg": np.full(n, 30.0),
    })
    df.to_csv(path, index=False)


def test_read_detections_sorts_by_track_then_frame(tmp_path: Path):
    p = tmp_path / "x.pylace_detections.csv"
    rows = [
        (5, 1, 50.0, 50.0),
        (3, 0, 30.0, 30.0),
        (4, 0, 40.0, 40.0),
        (3, 1, 31.0, 31.0),
    ]
    df = pd.DataFrame({
        "frame_idx": [r[0] for r in rows],
        "roi_label": ["_merged"] * 4,
        "track_id": [r[1] for r in rows],
        "cx_px": [r[2] for r in rows],
        "cy_px": [r[3] for r in rows],
        "x_mm": [0.0] * 4, "y_mm": [0.0] * 4,
        "area_px": [2500.0] * 4, "perimeter_px": [180.0] * 4,
        "solidity": [0.9] * 4,
        "major_axis_px": [58.0] * 4, "minor_axis_px": [22.0] * 4,
        "orientation_deg": [0.0] * 4,
    })
    df.to_csv(p, index=False)
    out = read_detections(p)
    pairs = list(zip(out["track_id"].tolist(), out["frame_idx"].tolist(), strict=True))
    assert pairs == [(0, 3), (0, 4), (1, 3), (1, 5)]


def test_read_detections_rejects_missing_columns(tmp_path: Path):
    p = tmp_path / "broken.csv"
    pd.DataFrame({"frame_idx": [0], "track_id": [0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        read_detections(p)


def test_write_trajectory_keeps_column_order(tmp_path: Path):
    p_in = tmp_path / "in.pylace_detections.csv"
    p_out = tmp_path / "out.csv"
    _write_minimal_detections(p_in)
    df = read_detections(p_in)
    df["cx_smooth_px"] = df["cx_px"]
    df["cy_smooth_px"] = df["cy_px"]
    df["interpolated"] = False
    write_trajectory(df, p_out)
    reloaded = pd.read_csv(p_out)
    cols = list(reloaded.columns)
    # Detection columns come first, in the canonical order.
    for i, name in enumerate(DETECTION_COLUMNS):
        assert cols[i] == name
    assert "cx_smooth_px" in cols
    assert "interpolated" in cols


def test_default_trajectory_path_replaces_detections_suffix():
    in_path = Path("/tmp/foo.mp4.pylace_detections.csv")
    out = default_trajectory_path(in_path)
    assert out.name == "foo.mp4.pylace_trajectory.csv"


def test_default_trajectory_path_handles_arbitrary_csv():
    in_path = Path("/tmp/random_name.csv")
    out = default_trajectory_path(in_path)
    assert out.name.endswith(".pylace_trajectory.csv")


def test_video_path_from_trajectory_strips_known_suffixes():
    from pylace.posthoc.io import video_path_from_trajectory

    cases = {
        "foo.mp4.pylace_detections.csv": "foo.mp4",
        "foo.mp4.pylace_trajectory.csv": "foo.mp4",
        "foo.mp4.pylace_audited.csv":    "foo.mp4",
        "random_name.csv":               "random_name",
    }
    for in_name, expected in cases.items():
        assert video_path_from_trajectory(Path("/tmp") / in_name).name == expected


def test_trajectory_stem_strips_known_suffixes():
    from pylace.posthoc.io import trajectory_stem

    cases = {
        "foo.mp4.pylace_detections.csv": "foo.mp4",
        "foo.mp4.pylace_trajectory.csv": "foo.mp4",
        "foo.mp4.pylace_audited.csv":    "foo.mp4",
    }
    for in_name, expected in cases.items():
        assert trajectory_stem(Path("/tmp") / in_name) == expected

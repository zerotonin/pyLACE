# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — tests.test_roi_sidecar                                 ║
# ║  « JSON round-trip + tolerant load + version handling »          ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Tests for the ROI sidecar JSON loader / writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pylace.annotator.geometry import Circle, Polygon, Rectangle
from pylace.roi.geometry import ROI, ROISet
from pylace.roi.sidecar import (
    SCHEMA_VERSION,
    ROISidecar,
    ROISidecarSchemaError,
    default_rois_path,
    read_rois,
    write_rois,
)


@pytest.fixture
def diverse_sidecar() -> ROISidecar:
    return ROISidecar(
        video_path="/tmp/example.mp4",
        video_sha256="0" * 64,
        roi_set=ROISet(
            rois=[
                ROI(shape=Circle(20.0, 30.0, 10.0), operation="add", label="centre"),
                ROI(
                    shape=Rectangle.from_two_points((0.0, 0.0), (100.0, 50.0)),
                    operation="subtract", label="top_strip",
                ),
                ROI(
                    shape=Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]),
                    operation="add", label="poly",
                ),
            ],
            mode="merge",
        ),
    )


def test_round_trip_preserves_all_rois(tmp_path: Path, diverse_sidecar: ROISidecar):
    out = tmp_path / "rois.json"
    write_rois(diverse_sidecar, out)
    loaded = read_rois(out)

    assert loaded.video_path == diverse_sidecar.video_path
    assert loaded.video_sha256 == diverse_sidecar.video_sha256
    assert loaded.roi_set.mode == "merge"
    assert len(loaded.roi_set.rois) == 3
    assert [r.label for r in loaded.roi_set.rois] == ["centre", "top_strip", "poly"]
    assert [r.operation for r in loaded.roi_set.rois] == ["add", "subtract", "add"]


def test_split_mode_round_trips_even_though_consumers_reject_it(
    tmp_path: Path,
):
    rs = ROISet(rois=[ROI(shape=Circle(0, 0, 1))], mode="split")
    sc = ROISidecar(video_path="x", video_sha256="0", roi_set=rs)
    out = tmp_path / "split.json"
    write_rois(sc, out)
    loaded = read_rois(out)
    assert loaded.roi_set.mode == "split"


def test_unknown_schema_version_raises(tmp_path: Path, diverse_sidecar: ROISidecar):
    out = tmp_path / "old.json"
    write_rois(diverse_sidecar, out)
    payload = json.loads(out.read_text())
    payload["schema_version"] = SCHEMA_VERSION + 99
    out.write_text(json.dumps(payload))
    with pytest.raises(ROISidecarSchemaError):
        read_rois(out)


def test_unknown_operation_raises(tmp_path: Path, diverse_sidecar: ROISidecar):
    out = tmp_path / "bad_op.json"
    write_rois(diverse_sidecar, out)
    payload = json.loads(out.read_text())
    payload["rois"][0]["operation"] = "intersect"
    out.write_text(json.dumps(payload))
    with pytest.raises(ROISidecarSchemaError):
        read_rois(out)


def test_empty_payload_loads_to_empty_roiset(tmp_path: Path):
    out = tmp_path / "empty.json"
    out.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "video": {"path": "/tmp/x.mp4", "sha256": "abc"},
        "mode": "merge",
        "rois": [],
    }))
    loaded = read_rois(out)
    assert loaded.roi_set.is_empty()


def test_default_rois_path_appends_suffix():
    video = Path("/data/recording_arena_03.mp4")
    assert default_rois_path(video).name == (
        "recording_arena_03.mp4.pylace_rois.json"
    )


def test_freehand_mask_round_trips_via_sibling_png(tmp_path: Path):
    import numpy as np

    fh = np.zeros((20, 30), dtype=np.uint8)
    fh[5:10, 5:10] = 255
    sc = ROISidecar(
        video_path="/tmp/x.mp4", video_sha256="0" * 64,
        roi_set=ROISet(
            rois=[ROI(shape=Circle(10.0, 10.0, 4.0))],
            freehand_mask=fh,
        ),
    )
    out = tmp_path / "rois.json"
    write_rois(sc, out)

    fh_path = out.with_suffix(".freehand.png")
    assert fh_path.exists()

    loaded = read_rois(out)
    assert loaded.roi_set.freehand_mask is not None
    assert loaded.roi_set.freehand_mask.shape == fh.shape
    assert (loaded.roi_set.freehand_mask > 0).sum() == (fh > 0).sum()


def test_save_without_freehand_clears_stale_sibling_png(tmp_path: Path):
    import numpy as np

    out = tmp_path / "rois.json"
    fh_path = out.with_suffix(".freehand.png")
    fh_path.write_bytes(b"\x89PNG stale stub")  # leave a stale file behind

    sc = ROISidecar(
        video_path="/tmp/x.mp4", video_sha256="0" * 64,
        roi_set=ROISet(rois=[ROI(shape=Circle(0, 0, 1))]),
    )
    write_rois(sc, out)
    assert not fh_path.exists()

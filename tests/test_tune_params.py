"""Tuning-params JSON round-trip + schema-version handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pylace.tune.params import (
    SCHEMA_VERSION,
    BackgroundParams,
    DetectionParams,
    TuningParams,
    TuningParamsSchemaError,
    default_params_path,
    read_params,
    write_params,
)


def test_defaults_match_detector_defaults():
    p = TuningParams.defaults()
    assert p.detection.threshold == 25
    assert p.detection.min_area == 20
    assert p.detection.max_area == 5000
    assert p.detection.morph_kernel == 3
    assert p.background.n_frames == 50
    assert p.background.start_frac == 0.1
    assert p.background.end_frac == 0.9


def test_params_round_trip(tmp_path: Path):
    out = tmp_path / "p.json"
    params = TuningParams(
        detection=DetectionParams(threshold=40, min_area=150, max_area=600, morph_kernel=5),
        background=BackgroundParams(n_frames=80, start_frac=0.25, end_frac=0.75),
    )
    write_params(params, video_path=Path("/tmp/x.mp4"), video_sha256_hex="a" * 64, out_path=out)

    loaded, video = read_params(out)
    assert loaded == params
    assert video["path"] == "/tmp/x.mp4"
    assert video["sha256"] == "a" * 64


def test_schema_version_mismatch_raises(tmp_path: Path):
    out = tmp_path / "p.json"
    write_params(
        TuningParams.defaults(),
        video_path=Path("/tmp/x.mp4"), video_sha256_hex="0" * 64, out_path=out,
    )
    payload = json.loads(out.read_text())
    payload["schema_version"] = SCHEMA_VERSION + 99
    out.write_text(json.dumps(payload))

    with pytest.raises(TuningParamsSchemaError):
        read_params(out)


def test_default_params_path_appends_suffix():
    video = Path("/data/recording_arena_03.mp4")
    assert default_params_path(video).name == (
        "recording_arena_03.mp4.pylace_detect_params.json"
    )

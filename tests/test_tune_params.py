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


def test_read_tolerant_of_missing_new_fields(tmp_path: Path):
    """Older sidecars without dilate_iters / erode_iters still load."""
    out = tmp_path / "old.json"
    out.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "video": {"path": "/tmp/x.mp4", "sha256": "0" * 64},
        "detection": {
            "threshold": 80, "min_area": 100, "max_area": 1500, "morph_kernel": 5,
        },
        "background": {"n_frames": 200, "start_frac": 0.1, "end_frac": 0.9},
    }))
    params, _ = read_params(out)
    assert params.detection.threshold == 80
    assert params.detection.dilate_iters == 0  # filled from dataclass default
    assert params.detection.erode_iters == 0


def test_tracking_params_default_to_enabled_with_sane_thresholds():
    p = TuningParams.defaults()
    assert p.tracking.enabled is True
    assert p.tracking.max_distance_px == 50.0
    assert p.tracking.max_missed_frames == 5


def test_tracking_params_round_trip(tmp_path: Path):
    from pylace.tune.params import TrackingParams

    out = tmp_path / "with_tracking.json"
    params = TuningParams(
        detection=DetectionParams(),
        background=BackgroundParams(),
        tracking=TrackingParams(
            enabled=False, max_distance_px=80.0, max_missed_frames=10,
            n_animals=3,
        ),
    )
    write_params(params, video_path=Path("/tmp/x.mp4"), video_sha256_hex="0" * 64,
                 out_path=out)
    loaded, _ = read_params(out)
    assert loaded.tracking.enabled is False
    assert loaded.tracking.max_distance_px == 80.0
    assert loaded.tracking.max_missed_frames == 10
    assert loaded.tracking.n_animals == 3


def test_tracking_params_n_animals_default_is_none():
    p = TuningParams.defaults()
    assert p.tracking.n_animals is None


def test_tracking_params_expected_animal_area_round_trip(tmp_path: Path):
    from pylace.tune.params import TrackingParams

    out = tmp_path / "with_area.json"
    params = TuningParams(
        detection=DetectionParams(),
        background=BackgroundParams(),
        tracking=TrackingParams(
            n_animals=3, expected_animal_area_px=2400.0,
        ),
    )
    write_params(params, video_path=Path("/tmp/x.mp4"), video_sha256_hex="0" * 64,
                 out_path=out)
    loaded, _ = read_params(out)
    assert loaded.tracking.expected_animal_area_px == 2400.0


def test_tracking_params_expected_animal_area_default_is_none():
    p = TuningParams.defaults()
    assert p.tracking.expected_animal_area_px is None


def test_old_sidecar_without_tracking_block_loads_with_defaults(tmp_path: Path):
    """Sidecars predating the tracking block fall back to defaults."""
    out = tmp_path / "legacy.json"
    out.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "video": {"path": "/tmp/x.mp4", "sha256": "0" * 64},
        "detection": {
            "threshold": 25, "min_area": 20, "max_area": 5000,
            "morph_kernel": 3, "dilate_iters": 0, "erode_iters": 0,
        },
        "background": {
            "n_frames": 50, "start_frac": 0.1, "end_frac": 0.9,
            "polarity": "dark_on_light",
        },
    }))
    loaded, _ = read_params(out)
    assert loaded.tracking.enabled is True
    assert loaded.tracking.max_distance_px == 50.0


def test_read_tolerant_of_unknown_fields(tmp_path: Path):
    """Future sidecars with extra keys do not break."""
    out = tmp_path / "future.json"
    out.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "video": {"path": "/tmp/x.mp4", "sha256": "0" * 64},
        "detection": {
            "threshold": 25, "min_area": 20, "max_area": 5000, "morph_kernel": 3,
            "dilate_iters": 0, "erode_iters": 0,
            "future_knob": "ignored",
        },
        "background": {"n_frames": 50, "start_frac": 0.1, "end_frac": 0.9},
    }))
    params, _ = read_params(out)
    assert params.detection.threshold == 25

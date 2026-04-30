"""JSON sidecar I/O for tuned detector and background parameters."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pylace.detect.frame import (
    DEFAULT_DILATE_ITERS,
    DEFAULT_ERODE_ITERS,
    DEFAULT_MAX_AREA,
    DEFAULT_MIN_AREA,
    DEFAULT_MORPH_KERNEL,
    DEFAULT_THRESHOLD,
)

Polarity = Literal["dark_on_light", "light_on_dark"]

SCHEMA_VERSION = 1
PARAMS_SUFFIX = ".pylace_detect_params.json"


class TuningParamsSchemaError(ValueError):
    """Raised when a tuning-params file cannot be parsed at this schema version."""


@dataclass
class DetectionParams:
    """Per-frame blob-detection parameters tunable in the GUI."""

    threshold: int = DEFAULT_THRESHOLD
    min_area: int = DEFAULT_MIN_AREA
    max_area: int = DEFAULT_MAX_AREA
    morph_kernel: int = DEFAULT_MORPH_KERNEL
    dilate_iters: int = DEFAULT_DILATE_ITERS
    erode_iters: int = DEFAULT_ERODE_ITERS


@dataclass
class BackgroundParams:
    """Background sampling parameters tunable in the GUI.

    ``polarity`` selects which projection feeds detection: dark animals
    on a bright arena use the max-projection; bright animals on a dark
    arena use the min-projection. Both projections are always computed
    and saved — the trail image (the "wrong" one for detection) shows
    where the animal lives most of the time and is useful for
    superimposing trajectories.
    """

    n_frames: int = 50
    start_frac: float = 0.1
    end_frac: float = 0.9
    polarity: Polarity = "dark_on_light"


@dataclass
class TuningParams:
    """Bundle of detection + background parameters saved as a sidecar."""

    detection: DetectionParams
    background: BackgroundParams

    @classmethod
    def defaults(cls) -> TuningParams:
        return cls(detection=DetectionParams(), background=BackgroundParams())


def write_params(
    params: TuningParams, video_path: Path, video_sha256_hex: str, out_path: Path,
) -> None:
    """Write tuning params to JSON, with the source video's identity stamped in."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "video": {"path": str(video_path), "sha256": video_sha256_hex},
        "detection": asdict(params.detection),
        "background": asdict(params.background),
    }
    out_path.write_text(json.dumps(payload, indent=2))


def read_params(in_path: Path) -> tuple[TuningParams, dict]:
    """Load tuning params; return ``(params, video_meta_dict)``.

    Tolerant of missing dataclass fields so old sidecars (pre-dilate/erode)
    still load by falling back to dataclass defaults.
    """
    payload = json.loads(in_path.read_text())
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise TuningParamsSchemaError(
            f"Unsupported tuning-params schema_version: {version!r} "
            f"(this build expects {SCHEMA_VERSION}).",
        )
    return (
        TuningParams(
            detection=_parse_dataclass(DetectionParams, payload["detection"]),
            background=_parse_dataclass(BackgroundParams, payload["background"]),
        ),
        payload.get("video", {}),
    )


def _parse_dataclass(cls, raw: dict):
    """Construct ``cls`` from a dict, ignoring unknown keys and using defaults."""
    valid = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in valid})


def default_params_path(video: Path) -> Path:
    """Conventional sidecar path for tuning params alongside the video."""
    return video.with_name(video.name + PARAMS_SUFFIX)

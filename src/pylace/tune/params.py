"""JSON sidecar I/O for tuned detector and background parameters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pylace.detect.frame import (
    DEFAULT_MAX_AREA,
    DEFAULT_MIN_AREA,
    DEFAULT_MORPH_KERNEL,
    DEFAULT_THRESHOLD,
)

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


@dataclass
class BackgroundParams:
    """Max-projection background sampling parameters tunable in the GUI."""

    n_frames: int = 50
    start_frac: float = 0.1
    end_frac: float = 0.9


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
    """Load tuning params; return ``(params, video_meta_dict)``."""
    payload = json.loads(in_path.read_text())
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise TuningParamsSchemaError(
            f"Unsupported tuning-params schema_version: {version!r} "
            f"(this build expects {SCHEMA_VERSION}).",
        )
    return (
        TuningParams(
            detection=DetectionParams(**payload["detection"]),
            background=BackgroundParams(**payload["background"]),
        ),
        payload.get("video", {}),
    )


def default_params_path(video: Path) -> Path:
    """Conventional sidecar path for tuning params alongside the video."""
    return video.with_name(video.name + PARAMS_SUFFIX)

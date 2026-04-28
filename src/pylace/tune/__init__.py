"""Interactive parameter tuning for the classical-CV detector."""

from pylace.tune.frame_loader import sample_preview_frames
from pylace.tune.overlay import render_overlay
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

__all__ = [
    "SCHEMA_VERSION",
    "BackgroundParams",
    "DetectionParams",
    "TuningParams",
    "TuningParamsSchemaError",
    "default_params_path",
    "read_params",
    "render_overlay",
    "sample_preview_frames",
    "write_params",
]

"""Per-video arena annotation: shape, world frame, pix-to-mm calibration."""

from pylace.annotator.geometry import (
    Arena,
    Calibration,
    Circle,
    Polygon,
    Rectangle,
    Vertex,
    WorldFrame,
    edge_length,
    pixel_to_world,
    shape_name,
)
from pylace.annotator.sidecar import (
    SCHEMA_VERSION,
    Sidecar,
    SidecarSchemaError,
    VideoMeta,
    default_sidecar_path,
    probe_video,
    read_sidecar,
    video_sha256,
    write_sidecar,
)

__all__ = [
    "SCHEMA_VERSION",
    "Arena",
    "Calibration",
    "Circle",
    "Polygon",
    "Rectangle",
    "Sidecar",
    "SidecarSchemaError",
    "Vertex",
    "VideoMeta",
    "WorldFrame",
    "default_sidecar_path",
    "edge_length",
    "pixel_to_world",
    "probe_video",
    "read_sidecar",
    "shape_name",
    "video_sha256",
    "write_sidecar",
]

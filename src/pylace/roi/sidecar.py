# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — roi.sidecar                                            ║
# ║  « JSON I/O for the per-video ROI set »                          ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Schema-version-1 sidecar at <video>.pylace_rois.json. Tolerant  ║
# ║  loader: missing keys fall back to defaults, unknown keys are    ║
# ║  ignored — older sidecars and forward-compatibility extensions   ║
# ║  both round-trip cleanly.                                        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""ROI sidecar JSON I/O."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pylace.annotator.geometry import Arena
from pylace.annotator.sidecar import _arena_from_dict, _arena_to_dict
from pylace.roi.constants import SCHEMA_VERSION, SIDECAR_SUFFIX
from pylace.roi.geometry import ROI, ROIMode, ROIOperation, ROISet


class ROISidecarSchemaError(ValueError):
    """Raised when an ROI sidecar cannot be parsed at the current schema."""


@dataclass
class ROISidecar:
    """Full ROI payload written next to a video."""

    video_path: str
    video_sha256: str
    roi_set: ROISet


def write_rois(sidecar: ROISidecar, out_path: Path) -> None:
    """Write an ROI sidecar to JSON; freehand mask goes to a sibling PNG.

    The PNG path is derived from the JSON path by replacing the final
    ``.json`` with ``.freehand.png``. If the ROI set has no freehand
    mask, any stale sibling PNG is removed.
    """
    out_path.write_text(json.dumps(_to_payload(sidecar), indent=2))
    fh_path = freehand_path_for_json(out_path)
    if sidecar.roi_set.has_freehand_mask():
        cv2.imwrite(str(fh_path), sidecar.roi_set.freehand_mask)
    elif fh_path.exists():
        fh_path.unlink()


def read_rois(in_path: Path) -> ROISidecar:
    """Load an ROI sidecar; raise if the schema version is unsupported.

    The sibling freehand PNG is loaded automatically when present.
    """
    payload = json.loads(in_path.read_text())
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ROISidecarSchemaError(
            f"Unsupported ROI schema_version: {version!r} "
            f"(this build expects {SCHEMA_VERSION}).",
        )
    sidecar = _from_payload(payload)
    fh_path = freehand_path_for_json(in_path)
    if fh_path.exists():
        loaded = cv2.imread(str(fh_path), cv2.IMREAD_GRAYSCALE)
        if loaded is not None:
            sidecar.roi_set.freehand_mask = loaded
    return sidecar


def default_rois_path(video: Path) -> Path:
    """Conventional sidecar path next to the video."""
    return video.with_name(video.name + SIDECAR_SUFFIX)


def freehand_path_for_json(json_path: Path) -> Path:
    """Sibling path of the freehand PNG given the ROI JSON path."""
    return json_path.with_suffix(".freehand.png")


# ─────────────────────────────────────────────────────────────────
#  Internal payload helpers
# ─────────────────────────────────────────────────────────────────

def _to_payload(sidecar: ROISidecar) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "video": {"path": sidecar.video_path, "sha256": sidecar.video_sha256},
        "mode": sidecar.roi_set.mode,
        "rois": [_roi_to_dict(r) for r in sidecar.roi_set.rois],
    }


def _from_payload(payload: dict) -> ROISidecar:
    video = payload.get("video", {})
    mode = payload.get("mode", "merge")
    if mode not in ("merge", "split"):
        raise ROISidecarSchemaError(f"Unknown ROI mode: {mode!r}.")
    roi_set = ROISet(
        rois=[_roi_from_dict(r) for r in payload.get("rois", [])],
        mode=mode,
    )
    return ROISidecar(
        video_path=str(video.get("path", "")),
        video_sha256=str(video.get("sha256", "")),
        roi_set=roi_set,
    )


def _roi_to_dict(roi: ROI) -> dict:
    return {
        "operation": roi.operation,
        "label": roi.label,
        "geometry": _arena_to_dict(roi.shape),
    }


def _roi_from_dict(raw: dict) -> ROI:
    operation = raw.get("operation", "add")
    if operation not in ("add", "subtract"):
        raise ROISidecarSchemaError(
            f"Unknown ROI operation: {operation!r}.",
        )
    geometry = raw.get("geometry")
    if geometry is None:
        raise ROISidecarSchemaError("ROI entry is missing 'geometry'.")
    shape: Arena = _arena_from_dict(geometry)
    return ROI(shape=shape, operation=operation, label=str(raw.get("label", "")))


__all__ = [
    "ROISidecar",
    "ROISidecarSchemaError",
    "default_rois_path",
    "read_rois",
    "write_rois",
]

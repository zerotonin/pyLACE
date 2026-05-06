# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — posthoc.io                                             ║
# ║  « detection CSV ↔ DataFrame; cleaned-trajectory CSV writer »    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""CSV I/O for the post-hoc trajectory cleaner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DETECTION_COLUMNS: tuple[str, ...] = (
    "frame_idx", "roi_label", "track_id",
    "cx_px", "cy_px", "x_mm", "y_mm",
    "area_px", "perimeter_px", "solidity",
    "major_axis_px", "minor_axis_px", "orientation_deg",
)

CLEANED_EXTRA_COLUMNS: tuple[str, ...] = (
    "cx_smooth_px", "cy_smooth_px",
    "vx_mm_s", "vy_mm_s", "speed_mm_s",
    "heading_deg", "yaw_deg", "yaw_rate_deg_s",
    "interpolated", "outlier_rejected", "heading_unresolved",
)

TRAJECTORY_SUFFIX = ".pylace_trajectory.csv"


def read_detections(csv_path: Path) -> pd.DataFrame:
    """Load a pylace-detect CSV and sort by (track_id, frame_idx)."""
    df = pd.read_csv(csv_path)
    missing = [c for c in DETECTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Detections CSV {csv_path} is missing columns: {missing}",
        )
    return df.sort_values(["track_id", "frame_idx"]).reset_index(drop=True)


def write_trajectory(df: pd.DataFrame, csv_path: Path) -> None:
    """Write a cleaned trajectory DataFrame to CSV with consistent column order."""
    cols = list(DETECTION_COLUMNS) + [
        c for c in CLEANED_EXTRA_COLUMNS if c in df.columns
    ]
    extra = [c for c in df.columns if c not in cols]
    cols.extend(extra)
    df.reindex(columns=cols).to_csv(csv_path, index=False, float_format="%.4f")


def default_trajectory_path(detections_csv: Path) -> Path:
    """Conventional output path next to the detections CSV."""
    name = detections_csv.name
    if name.endswith(".pylace_detections.csv"):
        stem = name[: -len(".pylace_detections.csv")]
        return detections_csv.with_name(stem + TRAJECTORY_SUFFIX)
    return detections_csv.with_name(detections_csv.stem + TRAJECTORY_SUFFIX)

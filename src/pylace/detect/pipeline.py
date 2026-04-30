"""Orchestrator: video + sidecar → per-frame detections → CSV rows."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pylace.annotator.geometry import pixel_to_world
from pylace.annotator.sidecar import Sidecar
from pylace.detect.arena_mask import arena_mask
from pylace.detect.background import build_max_projection_background
from pylace.detect.frame import (
    DEFAULT_DILATE_ITERS,
    DEFAULT_ERODE_ITERS,
    DEFAULT_MAX_AREA,
    DEFAULT_MIN_AREA,
    DEFAULT_MORPH_KERNEL,
    DEFAULT_THRESHOLD,
    Detection,
    detect_blobs,
)
from pylace.detect.chain import ChainSplitter
from pylace.tracking.tracks import Tracker

CSV_COLUMNS = (
    "frame_idx", "roi_label", "track_id",
    "cx_px", "cy_px", "x_mm", "y_mm",
    "area_px", "perimeter_px",
    "major_axis_px", "minor_axis_px", "orientation_deg",
)

MERGED_ROI_LABEL = "_merged"


@dataclass
class FrameResult:
    """Detections found in one source frame."""

    frame_idx: int
    detections: list[Detection]
    roi_label: str = MERGED_ROI_LABEL


def run_detection(
    video_path: Path,
    sidecar: Sidecar,
    *,
    threshold: int = DEFAULT_THRESHOLD,
    min_area: int = DEFAULT_MIN_AREA,
    max_area: int = DEFAULT_MAX_AREA,
    morph_kernel: int = DEFAULT_MORPH_KERNEL,
    dilate_iters: int = DEFAULT_DILATE_ITERS,
    erode_iters: int = DEFAULT_ERODE_ITERS,
    every: int = 1,
    max_frames: int | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    background: np.ndarray | None = None,
    extra_mask: np.ndarray | None = None,
    tracker: Tracker | None = None,
    chain_splitter: ChainSplitter | None = None,
) -> Iterator[FrameResult]:
    """Stream per-frame detections for a video against a sidecar.

    Args:
        video_path: Source video.
        sidecar: Arena geometry, world frame, and calibration.
        threshold: bg-fg intensity-difference threshold.
        min_area, max_area: Blob-area gates in pixels.
        morph_kernel: Side length of the open/close structuring element.
        every: Skip ahead this many frames between detections (1 = process
            every frame). Counted from ``start_frame``.
        max_frames: Stop after this many *processed* frames (not raw frames).
        start_frame: First frame index to process. Background sampling is
            independent of this — the default max-projection samples
            broadly across the video.
        end_frame: One past the last frame index to process; ``None`` means
            run to the end of the video.
        background: Precomputed background; if None, build one via
            max-projection from the video itself.

    Yields:
        ``FrameResult`` per processed frame, in source order. ``frame_idx``
        is the absolute frame index in the source video so rows remain
        traceable when ``start_frame`` is non-zero.
    """
    if every < 1:
        raise ValueError("every must be >= 1.")
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0.")
    if end_frame is not None and end_frame <= start_frame:
        raise ValueError("end_frame must be > start_frame.")

    bg = (
        background
        if background is not None
        else build_max_projection_background(video_path)
    )
    mask = arena_mask(sidecar.arena, sidecar.video.frame_size)
    if extra_mask is not None:
        if extra_mask.shape != mask.shape:
            raise ValueError(
                f"extra_mask shape {extra_mask.shape} does not match arena mask "
                f"{mask.shape}.",
            )
        mask = mask & extra_mask.astype(bool)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
    try:
        yield from _iter_frames(
            cap=cap, bg=bg, mask=mask,
            threshold=threshold, min_area=min_area, max_area=max_area,
            morph_kernel=morph_kernel,
            dilate_iters=dilate_iters, erode_iters=erode_iters,
            every=every, max_frames=max_frames,
            start_frame=start_frame, end_frame=end_frame,
            tracker=tracker, chain_splitter=chain_splitter,
        )
    finally:
        cap.release()


def _iter_frames(
    *, cap: cv2.VideoCapture, bg: np.ndarray, mask: np.ndarray,
    threshold: int, min_area: int, max_area: int, morph_kernel: int,
    dilate_iters: int, erode_iters: int,
    every: int, max_frames: int | None,
    start_frame: int, end_frame: int | None,
    tracker: Tracker | None,
    chain_splitter: ChainSplitter | None,
) -> Iterator[FrameResult]:
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    idx = start_frame
    kept = 0
    while True:
        if end_frame is not None and idx >= end_frame:
            break
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if (idx - start_frame) % every == 0:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            detections = detect_blobs(
                gray, bg, mask,
                threshold=threshold, min_area=min_area, max_area=max_area,
                morph_kernel=morph_kernel,
                dilate_iters=dilate_iters, erode_iters=erode_iters,
                keep_contour=False,
                chain_splitter=chain_splitter,
            )
            if tracker is not None:
                detections = tracker.step(idx, detections)
            yield FrameResult(frame_idx=idx, detections=detections)
            kept += 1
            if max_frames is not None and kept >= max_frames:
                break
        idx += 1


def write_detections_csv(
    results: Iterable[FrameResult],
    sidecar: Sidecar,
    out_path: Path,
) -> int:
    """Write per-frame per-detection rows to a CSV; return the row count."""
    n = 0
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for fr in results:
            for det_idx, d in enumerate(fr.detections):
                x_mm, y_mm = pixel_to_world(
                    (d.cx, d.cy), sidecar.world_frame, sidecar.calibration,
                )
                track_id = d.track_id if d.track_id >= 0 else det_idx
                writer.writerow([
                    fr.frame_idx, fr.roi_label, track_id,
                    f"{d.cx:.3f}", f"{d.cy:.3f}",
                    f"{x_mm:.4f}", f"{y_mm:.4f}",
                    f"{d.area_px:.1f}", f"{d.perimeter_px:.2f}",
                    f"{d.major_axis_px:.3f}", f"{d.minor_axis_px:.3f}",
                    f"{d.orientation_deg:.2f}",
                ])
                n += 1
    return n

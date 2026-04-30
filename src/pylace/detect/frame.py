"""Per-frame detection: bg-sub → threshold → morphology → contour → ellipse."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_THRESHOLD = 25
DEFAULT_MIN_AREA = 20
DEFAULT_MAX_AREA = 5000
DEFAULT_MORPH_KERNEL = 3
DEFAULT_DILATE_ITERS = 0
DEFAULT_ERODE_ITERS = 0
DEFAULT_MIN_SOLIDITY = 0.0          # 0 = disabled (no minimum)
DEFAULT_MAX_AXIS_RATIO = 0.0        # 0 = disabled (no maximum)
ELLIPSE_FIT_MIN_POINTS = 5


@dataclass
class Detection:
    """One blob's per-frame summary in image (pixel) coordinates.

    ``track_id`` is set in place by :class:`pylace.tracking.Tracker`
    after detection. The sentinel ``-1`` means "no tracking applied";
    the CSV writer treats that as "use the per-frame index instead".

    ``perimeter_px`` is the contour's arc length, used by the Hungarian
    cost matrix when contour-similarity weighting is enabled.
    """

    cx: float
    cy: float
    area_px: float
    perimeter_px: float
    solidity: float
    major_axis_px: float
    minor_axis_px: float
    orientation_deg: float
    contour: np.ndarray | None = None
    track_id: int = -1


def detect_blobs(
    frame_gray: np.ndarray,
    background_gray: np.ndarray,
    arena_mask: np.ndarray,
    *,
    threshold: int = DEFAULT_THRESHOLD,
    min_area: int = DEFAULT_MIN_AREA,
    max_area: int = DEFAULT_MAX_AREA,
    morph_kernel: int = DEFAULT_MORPH_KERNEL,
    dilate_iters: int = DEFAULT_DILATE_ITERS,
    erode_iters: int = DEFAULT_ERODE_ITERS,
    min_solidity: float = DEFAULT_MIN_SOLIDITY,
    max_axis_ratio: float = DEFAULT_MAX_AXIS_RATIO,
    keep_contour: bool = True,
    chain_splitter=None,
) -> list[Detection]:
    """Per-frame blob detection on a grayscale frame.

    Dark animals on a bright background — the foreground mask is
    ``background - frame > threshold`` after morphological cleanup, masked
    to the arena interior.

    Args:
        frame_gray: ``uint8`` grayscale frame.
        background_gray: Reference background, same shape and dtype.
        arena_mask: Boolean mask, True inside the arena, same shape.
        threshold: Foreground intensity-difference threshold.
        min_area: Reject blobs smaller than this (in pixels).
        max_area: Reject blobs larger than this (in pixels).
        morph_kernel: Side length of the elliptical structuring element used
            for opening and closing.
        dilate_iters: Extra dilation iterations applied after open+close.
            Useful for merging fragmented blobs (e.g. one fly split by a
            shadow into two pieces).
        erode_iters: Extra erosion iterations applied after dilation.
            Useful for shrinking the blob back if dilation over-grew, or
            for trimming peripheral noise.
        keep_contour: If True, return the raw contour with each detection;
            otherwise leave ``Detection.contour`` as None to save memory.
    """
    detections, _ = detect_blobs_with_mask(
        frame_gray, background_gray, arena_mask,
        threshold=threshold, min_area=min_area, max_area=max_area,
        morph_kernel=morph_kernel,
        dilate_iters=dilate_iters, erode_iters=erode_iters,
        min_solidity=min_solidity, max_axis_ratio=max_axis_ratio,
        keep_contour=keep_contour,
        chain_splitter=chain_splitter,
    )
    return detections


def detect_blobs_with_mask(
    frame_gray: np.ndarray,
    background_gray: np.ndarray,
    arena_mask: np.ndarray,
    *,
    threshold: int = DEFAULT_THRESHOLD,
    min_area: int = DEFAULT_MIN_AREA,
    max_area: int = DEFAULT_MAX_AREA,
    morph_kernel: int = DEFAULT_MORPH_KERNEL,
    dilate_iters: int = DEFAULT_DILATE_ITERS,
    erode_iters: int = DEFAULT_ERODE_ITERS,
    min_solidity: float = DEFAULT_MIN_SOLIDITY,
    max_axis_ratio: float = DEFAULT_MAX_AXIS_RATIO,
    keep_contour: bool = True,
    chain_splitter=None,
) -> tuple[list[Detection], np.ndarray]:
    """Like :func:`detect_blobs` but also returns the binary foreground mask.

    The mask is the post-morphology, arena-clipped foreground used for
    contour finding — useful for tuning UIs that want to show which pixels
    survived the threshold + open/close pipeline before any contour was
    fitted.

    Returns:
        ``(detections, foreground_mask)`` where ``foreground_mask`` is a
        ``uint8`` 0/255 array the same shape as the frame.
    """
    fg = _foreground_mask(
        frame_gray, background_gray, arena_mask,
        threshold=threshold, morph_kernel=morph_kernel,
        dilate_iters=dilate_iters, erode_iters=erode_iters,
    )
    contours = _filtered_contours(fg, min_area, max_area)
    if chain_splitter is not None:
        contours = chain_splitter.maybe_split(contours)
    out: list[Detection] = []
    for c in contours:
        if len(c) < ELLIPSE_FIT_MIN_POINTS:
            continue
        det = _contour_to_detection(c, keep_contour=keep_contour)
        if min_solidity > 0.0 and det.solidity < min_solidity:
            continue
        if max_axis_ratio > 0.0 and det.minor_axis_px > 0.0:
            ratio = det.major_axis_px / det.minor_axis_px
            if ratio > max_axis_ratio:
                continue
        out.append(det)
    return out, fg


def _foreground_mask(
    frame: np.ndarray, bg: np.ndarray, mask: np.ndarray,
    *, threshold: int, morph_kernel: int,
    dilate_iters: int, erode_iters: int,
) -> np.ndarray:
    diff = cv2.subtract(bg, frame)
    fg = (diff > threshold).astype(np.uint8) * 255
    if morph_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel),
        )
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        if dilate_iters > 0:
            fg = cv2.dilate(fg, kernel, iterations=dilate_iters)
        if erode_iters > 0:
            fg = cv2.erode(fg, kernel, iterations=erode_iters)
    fg = cv2.bitwise_and(fg, mask.astype(np.uint8) * 255)
    return fg


def _filtered_contours(
    fg: np.ndarray, min_area: int, max_area: int,
) -> list[np.ndarray]:
    contours, _ = cv2.findContours(
        fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )
    return [
        c for c in contours
        if min_area <= cv2.contourArea(c) <= max_area
    ]


def _contour_to_detection(c: np.ndarray, *, keep_contour: bool) -> Detection:
    (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(c)
    major = max(axis_a, axis_b)
    minor = min(axis_a, axis_b)
    area = float(cv2.contourArea(c))
    hull_area = float(cv2.contourArea(cv2.convexHull(c)))
    solidity = area / hull_area if hull_area > 0 else 1.0
    return Detection(
        cx=float(cx),
        cy=float(cy),
        area_px=area,
        perimeter_px=float(cv2.arcLength(c, closed=True)),
        solidity=solidity,
        major_axis_px=float(major),
        minor_axis_px=float(minor),
        orientation_deg=float(angle),
        contour=c if keep_contour else None,
    )

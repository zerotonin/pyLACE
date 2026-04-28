"""Per-frame detection: bg-sub → threshold → morphology → contour → ellipse."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_THRESHOLD = 25
DEFAULT_MIN_AREA = 20
DEFAULT_MAX_AREA = 5000
DEFAULT_MORPH_KERNEL = 3
ELLIPSE_FIT_MIN_POINTS = 5


@dataclass
class Detection:
    """One blob's per-frame summary in image (pixel) coordinates."""

    cx: float
    cy: float
    area_px: float
    major_axis_px: float
    minor_axis_px: float
    orientation_deg: float
    contour: np.ndarray | None = None


def detect_blobs(
    frame_gray: np.ndarray,
    background_gray: np.ndarray,
    arena_mask: np.ndarray,
    *,
    threshold: int = DEFAULT_THRESHOLD,
    min_area: int = DEFAULT_MIN_AREA,
    max_area: int = DEFAULT_MAX_AREA,
    morph_kernel: int = DEFAULT_MORPH_KERNEL,
    keep_contour: bool = True,
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
        keep_contour: If True, return the raw contour with each detection;
            otherwise leave ``Detection.contour`` as None to save memory.
    """
    detections, _ = detect_blobs_with_mask(
        frame_gray, background_gray, arena_mask,
        threshold=threshold, min_area=min_area, max_area=max_area,
        morph_kernel=morph_kernel, keep_contour=keep_contour,
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
    keep_contour: bool = True,
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
        frame_gray, background_gray, arena_mask, threshold, morph_kernel,
    )
    contours = _filtered_contours(fg, min_area, max_area)
    out: list[Detection] = []
    for c in contours:
        if len(c) < ELLIPSE_FIT_MIN_POINTS:
            continue
        out.append(_contour_to_detection(c, keep_contour=keep_contour))
    return out, fg


def _foreground_mask(
    frame: np.ndarray, bg: np.ndarray, mask: np.ndarray,
    threshold: int, morph_kernel: int,
) -> np.ndarray:
    diff = cv2.subtract(bg, frame)
    fg = (diff > threshold).astype(np.uint8) * 255
    if morph_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel),
        )
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
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
    return Detection(
        cx=float(cx),
        cy=float(cy),
        area_px=float(cv2.contourArea(c)),
        major_axis_px=float(major),
        minor_axis_px=float(minor),
        orientation_deg=float(angle),
        contour=c if keep_contour else None,
    )

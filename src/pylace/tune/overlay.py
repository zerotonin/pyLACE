"""Render arena + detection overlays onto a grayscale frame for display."""

from __future__ import annotations

import cv2
import numpy as np

from pylace.annotator.geometry import Arena, Circle
from pylace.detect.frame import Detection
from pylace.inspect.traces import render_roi_outlines as _render_roi_outlines
from pylace.roi.geometry import ROISet

ARENA_COLOR = (0, 220, 220)        # cyan
CONTOUR_COLOR = (0, 255, 0)        # green
ELLIPSE_COLOR = (0, 165, 255)      # orange
CENTROID_COLOR = (0, 0, 255)       # red
NUMBER_COLOR = (255, 255, 255)     # white (with black outline for legibility)
MASK_TINT = (60, 0, 0)             # subtle blue tint on foreground pixels


def render_overlay(
    frame_gray: np.ndarray,
    arena: Arena,
    detections: list[Detection],
    *,
    foreground_mask: np.ndarray | None = None,
    roi_set: ROISet | None = None,
    show_arena: bool = True,
    show_roi: bool = False,
    show_contours: bool = True,
    show_ellipses: bool = True,
    show_centroids: bool = True,
    show_numbers: bool = False,
) -> np.ndarray:
    """Return a BGR image with overlays drawn for display.

    Args:
        frame_gray: ``uint8`` grayscale frame.
        arena: Arena geometry for the cyan boundary outline.
        detections: List of per-frame ``Detection`` objects.
        foreground_mask: Optional ``uint8``/``bool`` mask the size of the
            frame; if provided, foreground pixels get a subtle tint to
            reveal which pixels survived threshold + morphology before
            contour fitting.
        roi_set: Optional ``ROISet`` whose outlines are drawn when
            ``show_roi`` is True (green for add ROIs, red for
            subtract, teal for the freehand mask), reusing the same
            helper the inspector uses so the look is consistent.
        show_arena, show_roi, show_contours, show_ellipses,
        show_centroids, show_numbers: Toggles. ``show_numbers``
            labels each detection with its track_id (if assigned)
            or per-frame index.

    Returns:
        ``(H, W, 3)`` BGR ``uint8`` image.
    """
    bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    if foreground_mask is not None:
        _tint_mask(bgr, foreground_mask, MASK_TINT)
    if show_arena:
        _draw_arena(bgr, arena)
    if show_roi and roi_set is not None and not roi_set.is_empty():
        _render_roi_outlines(bgr, roi_set)
    for idx, d in enumerate(detections):
        if show_contours and d.contour is not None:
            cv2.drawContours(bgr, [d.contour], -1, CONTOUR_COLOR, 1)
        if show_ellipses:
            _draw_detection_ellipse(bgr, d)
        if show_centroids:
            cv2.drawMarker(
                bgr, (int(round(d.cx)), int(round(d.cy))),
                CENTROID_COLOR, cv2.MARKER_CROSS, 8, 1,
            )
        if show_numbers:
            _draw_detection_number(bgr, d, idx)
    return bgr


def _tint_mask(bgr: np.ndarray, mask: np.ndarray, tint_bgr: tuple[int, int, int]) -> None:
    bool_mask = mask.astype(bool)
    if not bool_mask.any():
        return
    tint = np.array(tint_bgr, dtype=np.int16)
    bgr[bool_mask] = np.clip(
        bgr[bool_mask].astype(np.int16) + tint, 0, 255,
    ).astype(np.uint8)


def _draw_arena(bgr: np.ndarray, arena: Arena) -> None:
    if isinstance(arena, Circle):
        cv2.circle(
            bgr, (int(round(arena.cx)), int(round(arena.cy))),
            int(round(arena.r)), ARENA_COLOR, 1,
        )
        return
    poly = np.array(arena.vertices, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(bgr, [poly], isClosed=True, color=ARENA_COLOR, thickness=1)


def _draw_detection_ellipse(bgr: np.ndarray, d: Detection) -> None:
    cv2.ellipse(
        bgr,
        (
            (float(d.cx), float(d.cy)),
            (float(d.minor_axis_px), float(d.major_axis_px)),
            float(d.orientation_deg),
        ),
        ELLIPSE_COLOR,
        1,
    )


def _draw_detection_number(bgr: np.ndarray, d: Detection, idx: int) -> None:
    """Label a detection with its track_id (if assigned) or per-frame index."""
    label = f"#{d.track_id}" if d.track_id >= 0 else f"#{idx}"
    pos = (int(round(d.cx)) + 6, int(round(d.cy)) - 6)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    cv2.putText(bgr, label, pos, font, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(bgr, label, pos, font, scale, NUMBER_COLOR, 1, cv2.LINE_AA)

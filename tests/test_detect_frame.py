"""Per-frame blob detection on synthetic frames."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pylace.annotator.geometry import Circle
from pylace.detect.arena_mask import arena_mask
from pylace.detect.frame import detect_blobs


def _bright_background(h: int = 100, w: int = 100, level: int = 220) -> np.ndarray:
    return np.full((h, w), level, dtype=np.uint8)


def _frame_with_dark_ellipse(
    h: int, w: int, *,
    cx: float, cy: float,
    axes: tuple[int, int],
    angle_deg: float,
    bg_level: int = 220,
    fg_level: int = 30,
) -> np.ndarray:
    frame = np.full((h, w), bg_level, dtype=np.uint8)
    cv2.ellipse(
        frame,
        (int(cx), int(cy)), axes,
        angle_deg, 0, 360,
        fg_level, thickness=-1,
    )
    return frame


def test_detect_finds_single_dark_ellipse_centroid_within_2px():
    h, w = 100, 100
    bg = _bright_background(h, w)
    frame = _frame_with_dark_ellipse(h, w, cx=50, cy=50, axes=(12, 5), angle_deg=30)
    mask = arena_mask(Circle(50.0, 50.0, 45.0), frame_size=(w, h))

    detections = detect_blobs(frame, bg, mask)

    assert len(detections) == 1
    d = detections[0]
    assert abs(d.cx - 50.0) < 2.0
    assert abs(d.cy - 50.0) < 2.0
    assert d.major_axis_px > d.minor_axis_px


def test_detect_finds_two_blobs_when_two_present():
    h, w = 100, 100
    bg = _bright_background(h, w)
    frame = bg.copy()
    cv2.ellipse(frame, (30, 30), (8, 4), 0, 0, 360, 30, thickness=-1)
    cv2.ellipse(frame, (70, 70), (8, 4), 90, 0, 360, 30, thickness=-1)
    mask = arena_mask(Circle(50.0, 50.0, 45.0), frame_size=(w, h))

    detections = detect_blobs(frame, bg, mask)

    centres = sorted((d.cx, d.cy) for d in detections)
    assert len(centres) == 2
    assert abs(centres[0][0] - 30) < 3
    assert abs(centres[1][0] - 70) < 3


def test_detect_ignores_blob_outside_arena_mask():
    h, w = 100, 100
    bg = _bright_background(h, w)
    frame = _frame_with_dark_ellipse(h, w, cx=10, cy=10, axes=(8, 4), angle_deg=0)
    mask = arena_mask(Circle(70.0, 70.0, 20.0), frame_size=(w, h))

    detections = detect_blobs(frame, bg, mask)

    assert detections == []


def test_detect_respects_min_area():
    h, w = 100, 100
    bg = _bright_background(h, w)
    frame = bg.copy()
    cv2.ellipse(frame, (50, 50), (3, 2), 0, 0, 360, 30, thickness=-1)
    mask = arena_mask(Circle(50.0, 50.0, 45.0), frame_size=(w, h))

    assert detect_blobs(frame, bg, mask, min_area=200) == []


def test_detect_respects_max_area():
    h, w = 100, 100
    bg = _bright_background(h, w)
    frame = bg.copy()
    cv2.ellipse(frame, (50, 50), (30, 20), 0, 0, 360, 30, thickness=-1)
    mask = arena_mask(Circle(50.0, 50.0, 45.0), frame_size=(w, h))

    assert detect_blobs(frame, bg, mask, max_area=50) == []


def test_detect_keep_contour_flag_drops_contour_when_false():
    h, w = 100, 100
    bg = _bright_background(h, w)
    frame = _frame_with_dark_ellipse(h, w, cx=50, cy=50, axes=(12, 5), angle_deg=0)
    mask = arena_mask(Circle(50.0, 50.0, 45.0), frame_size=(w, h))

    with_c = detect_blobs(frame, bg, mask, keep_contour=True)
    without_c = detect_blobs(frame, bg, mask, keep_contour=False)

    assert with_c[0].contour is not None
    assert without_c[0].contour is None


def test_dilate_iters_increases_blob_area():
    h, w = 100, 100
    bg = _bright_background(h, w)
    frame = _frame_with_dark_ellipse(h, w, cx=50, cy=50, axes=(8, 4), angle_deg=0)
    mask = arena_mask(Circle(50.0, 50.0, 45.0), frame_size=(w, h))

    base = detect_blobs(frame, bg, mask)
    dilated = detect_blobs(frame, bg, mask, dilate_iters=3)

    assert len(base) == len(dilated) == 1
    assert dilated[0].area_px > base[0].area_px


def test_erode_iters_decreases_blob_area():
    h, w = 100, 100
    bg = _bright_background(h, w)
    frame = _frame_with_dark_ellipse(h, w, cx=50, cy=50, axes=(10, 6), angle_deg=0)
    mask = arena_mask(Circle(50.0, 50.0, 45.0), frame_size=(w, h))

    base = detect_blobs(frame, bg, mask)
    eroded = detect_blobs(frame, bg, mask, erode_iters=2)

    assert len(base) == 1
    if eroded:
        assert eroded[0].area_px < base[0].area_px


def test_dilate_can_merge_two_close_blobs():
    h, w = 80, 80
    bg = _bright_background(h, w)
    frame = bg.copy()
    cv2.ellipse(frame, (35, 40), (4, 3), 0, 0, 360, 30, thickness=-1)
    cv2.ellipse(frame, (45, 40), (4, 3), 0, 0, 360, 30, thickness=-1)
    mask = arena_mask(Circle(40.0, 40.0, 35.0), frame_size=(w, h))

    base = detect_blobs(frame, bg, mask, morph_kernel=3)
    merged = detect_blobs(frame, bg, mask, morph_kernel=3, dilate_iters=4)

    assert len(base) >= 1
    assert len(merged) == 1

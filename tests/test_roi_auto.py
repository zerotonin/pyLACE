# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — tests.test_roi_auto                                    ║
# ║  « auto-ROI generation from the bg-pair difference »             ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Tests for :func:`pylace.roi.auto_roi.auto_rois_from_diff`."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pylace.annotator.geometry import Circle, Polygon
from pylace.roi.auto_roi import AutoRoiParams, auto_rois_from_diff


def _bg_with_blobs(blobs: list[tuple[int, int, int]], size: int = 80) -> np.ndarray:
    """Grayscale background of size×size with one bright disc per (cx, cy, r)."""
    img = np.zeros((size, size), dtype=np.uint8)
    for cx, cy, r in blobs:
        cv2.circle(img, (cx, cy), r, 220, thickness=-1)
    return img


def _full_arena_mask(size: int = 80) -> np.ndarray:
    return np.ones((size, size), dtype=bool)


def test_auto_roi_finds_one_polygon_for_one_trail_blob():
    bg_a = _bg_with_blobs([(40, 40, 8)])  # animal sat here
    bg_b = np.zeros_like(bg_a)            # never bright when animal absent
    rois = auto_rois_from_diff(bg_a, bg_b, _full_arena_mask())
    assert len(rois) == 1
    assert isinstance(rois[0].shape, Polygon)
    assert rois[0].operation == "add"
    assert rois[0].label == "auto"


def test_auto_roi_finds_one_polygon_per_disjoint_trail_blob():
    bg_a = _bg_with_blobs([(20, 20, 6), (60, 60, 6)])
    bg_b = np.zeros_like(bg_a)
    rois = auto_rois_from_diff(bg_a, bg_b, _full_arena_mask())
    assert len(rois) == 2


def test_auto_roi_polygon_is_simplified_not_per_pixel():
    bg_a = _bg_with_blobs([(40, 40, 12)])
    bg_b = np.zeros_like(bg_a)
    rois = auto_rois_from_diff(bg_a, bg_b, _full_arena_mask())
    poly = rois[0].shape
    assert isinstance(poly, Polygon)
    # A 12-px-radius disc would yield ~75 contour points raw; the
    # default Douglas-Peucker simplification should keep this much
    # smaller while remaining a valid polygon.
    assert 3 <= len(poly.vertices) < 30


def test_auto_roi_skips_blobs_below_min_area():
    bg_a = _bg_with_blobs([(40, 40, 1)])  # tiny — area < 100 px²
    bg_b = np.zeros_like(bg_a)
    rois = auto_rois_from_diff(bg_a, bg_b, _full_arena_mask())
    assert rois == []


def test_auto_roi_dilates_so_polygon_is_larger_than_raw_trail():
    # Tight blob with a small dilate → polygon diameter > raw blob diameter.
    bg_a = _bg_with_blobs([(40, 40, 6)])
    bg_b = np.zeros_like(bg_a)
    raw_area = float(((bg_a > 30) & (bg_a > 0)).sum())
    rois = auto_rois_from_diff(
        bg_a, bg_b, _full_arena_mask(),
        params=AutoRoiParams(dilate_iters=4, erode_iters=0),
    )
    assert len(rois) == 1
    poly_pts = np.array(rois[0].shape.vertices, dtype=np.int32)
    poly_area = float(cv2.contourArea(poly_pts))
    assert poly_area > raw_area


def test_auto_roi_clips_to_arena_mask():
    bg_a = _bg_with_blobs([(40, 40, 8), (10, 10, 6)])
    bg_b = np.zeros_like(bg_a)
    arena = np.zeros_like(bg_a, dtype=bool)
    cv2.circle(arena.view(np.uint8), (40, 40), 25, 1, thickness=-1)
    arena = arena.astype(bool)
    rois = auto_rois_from_diff(bg_a, bg_b, arena)
    # The corner blob is outside the arena and should be excluded.
    assert len(rois) == 1


def test_auto_roi_rejects_shape_mismatch():
    bg_a = np.zeros((20, 20), dtype=np.uint8)
    bg_b = np.zeros((30, 30), dtype=np.uint8)
    arena = np.ones((20, 20), dtype=bool)
    with pytest.raises(ValueError):
        auto_rois_from_diff(bg_a, bg_b, arena)


def test_auto_roi_returns_empty_when_bgs_are_equal():
    bg = _bg_with_blobs([(40, 40, 8)])
    rois = auto_rois_from_diff(bg, bg, _full_arena_mask())
    assert rois == []

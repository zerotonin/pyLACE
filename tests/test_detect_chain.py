# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — tests.test_detect_chain                                ║
# ║  « ChainSplitter + perpendicular-axis split helper »             ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Tests for :mod:`pylace.detect.chain`."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pylace.detect.chain import ChainSplitter, split_along_major_axis


def _filled_ellipse_contour(
    *, cx: int, cy: int, axes: tuple[int, int], angle: int, image_size: int = 200,
) -> np.ndarray:
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, 255, thickness=-1)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert len(contours) == 1
    return contours[0]


def _two_kissing_ellipses_contour(
    *, gap_px: int = 0, image_size: int = 200,
) -> np.ndarray:
    """Filled blob made of two horizontally-adjacent ellipses (a chain)."""
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.ellipse(img, (image_size // 2 - 20 - gap_px, image_size // 2),
                (20, 8), 0, 0, 360, 255, thickness=-1)
    cv2.ellipse(img, (image_size // 2 + 20 + gap_px, image_size // 2),
                (20, 8), 0, 0, 360, 255, thickness=-1)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert len(contours) == 1
    return contours[0]


# ─────────────────────────────────────────────────────────────────
#  split_along_major_axis
# ─────────────────────────────────────────────────────────────────


def test_split_returns_two_halves_for_chained_blob():
    contour = _two_kissing_ellipses_contour()
    halves = split_along_major_axis(contour)
    assert len(halves) == 2
    # Each half should still have enough points to fit an ellipse.
    for half in halves:
        assert len(half) >= 5


def test_split_halves_have_centroids_on_opposite_sides_of_original():
    contour = _two_kissing_ellipses_contour()
    halves = split_along_major_axis(contour)
    moments_orig = cv2.moments(contour)
    cx_orig = moments_orig["m10"] / moments_orig["m00"]

    centroids_x = []
    for half in halves:
        m = cv2.moments(half)
        centroids_x.append(m["m10"] / m["m00"])

    assert (centroids_x[0] - cx_orig) * (centroids_x[1] - cx_orig) < 0


def test_split_returns_empty_for_too_small_contour():
    pts = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.int32).reshape(-1, 1, 2)
    assert split_along_major_axis(pts) == []


def test_split_handles_vertical_chain():
    """A vertical kissing-ellipses blob still splits into top + bottom."""
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.ellipse(img, (100, 80), (8, 20), 0, 0, 360, 255, thickness=-1)
    cv2.ellipse(img, (100, 120), (8, 20), 0, 0, 360, 255, thickness=-1)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    halves = split_along_major_axis(contours[0])
    assert len(halves) == 2
    centroids_y = []
    for half in halves:
        m = cv2.moments(half)
        centroids_y.append(m["m01"] / m["m00"])
    # Halves are on opposite sides of the original centroid.
    assert (centroids_y[0] - 100) * (centroids_y[1] - 100) < 0


# ─────────────────────────────────────────────────────────────────
#  ChainSplitter
# ─────────────────────────────────────────────────────────────────


def test_splitter_passes_through_when_below_threshold():
    splitter = ChainSplitter(expected_animal_area_px=1000.0)
    small = _filled_ellipse_contour(cx=100, cy=100, axes=(20, 8), angle=0)
    out = splitter.maybe_split([small])
    assert len(out) == 1
    assert out[0] is small


def test_splitter_splits_when_oversized():
    splitter = ChainSplitter(
        expected_animal_area_px=400.0, area_ratio_threshold=1.5,
    )
    chain = _two_kissing_ellipses_contour()
    out = splitter.maybe_split([chain])
    assert len(out) == 2


def test_splitter_falls_back_to_original_when_split_fails():
    """A contour with too few points returns unchanged even though oversized."""
    splitter = ChainSplitter(expected_animal_area_px=1.0)
    pts = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.int32).reshape(-1, 1, 2)
    out = splitter.maybe_split([pts])
    assert len(out) == 1


def test_splitter_auto_learn_locks_expected_area_after_K_frames():
    """During learning the splitter is a no-op; afterwards it splits."""
    splitter = ChainSplitter(
        expected_animal_area_px=None,
        learn_frames=5, area_ratio_threshold=1.5,
    )
    single = _filled_ellipse_contour(cx=100, cy=100, axes=(20, 8), angle=0)
    chain = _two_kissing_ellipses_contour()

    # First 5 frames are pure observation; nothing splits.
    for _ in range(5):
        out = splitter.maybe_split([single])
        assert out == [single]
    assert splitter.is_learning is False
    assert splitter.expected_animal_area_px is not None

    out = splitter.maybe_split([chain])
    assert len(out) == 2


def test_splitter_rejects_invalid_args():
    with pytest.raises(ValueError):
        ChainSplitter(area_ratio_threshold=1.0)
    with pytest.raises(ValueError):
        ChainSplitter(learn_frames=0)
    with pytest.raises(ValueError):
        ChainSplitter(expected_animal_area_px=0)


def test_splitter_exposes_learned_area_on_property():
    splitter = ChainSplitter(expected_animal_area_px=None, learn_frames=3)
    contour = _filled_ellipse_contour(cx=100, cy=100, axes=(20, 8), angle=0)
    for _ in range(3):
        splitter.maybe_split([contour])
    assert splitter.expected_animal_area_px is not None
    # Area of a (20,8) ellipse ≈ π·20·8 ≈ 502.65; cv2's contour area is
    # close but not identical due to discretisation.
    assert 400 < splitter.expected_animal_area_px < 600

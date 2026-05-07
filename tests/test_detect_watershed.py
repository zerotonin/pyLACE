"""Watershed-based splitter for merged-fly blobs."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pylace.detect.watershed import WatershedSplitter, watershed_split


def _two_disk_blob(w: int = 200, h: int = 200,
                   c1: tuple[int, int] = (75, 100),
                   c2: tuple[int, int] = (115, 100),
                   r: int = 28) -> tuple[np.ndarray, np.ndarray]:
    """Two overlapping disks merged into a single contour. Returns (mask, contour)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, c1, r, 255, thickness=-1)
    cv2.circle(mask, c2, r, 255, thickness=-1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert len(contours) == 1, "the two disks must merge into one contour"
    return mask, contours[0]


def _three_disk_blob() -> tuple[np.ndarray, np.ndarray]:
    """Triangular cluster of three touching disks."""
    h, w = 220, 220
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (90, 80), 28, 255, thickness=-1)
    cv2.circle(mask, (130, 80), 28, 255, thickness=-1)
    cv2.circle(mask, (110, 115), 28, 255, thickness=-1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert len(contours) == 1
    return mask, contours[0]


# ─────────────────────────────────────────────────────────────────
#  Pure function: watershed_split
# ─────────────────────────────────────────────────────────────────


def test_watershed_split_two_disks_yields_two_contours():
    mask, contour = _two_disk_blob()
    out = watershed_split(contour, mask, peak_min_distance_px=8)
    assert len(out) == 2
    # Each output contour should have area roughly half of the merged
    # blob (within a generous tolerance — watershed lines are 1 px wide
    # so a few pixels are lost).
    total = float(cv2.contourArea(contour))
    for c in out:
        assert 0.3 * total < cv2.contourArea(c) < 0.7 * total


def test_watershed_split_three_disks_yields_three_contours():
    mask, contour = _three_disk_blob()
    out = watershed_split(contour, mask, peak_min_distance_px=8)
    assert len(out) == 3


def test_watershed_split_single_disk_returns_empty():
    """A single round blob has only one peak; can't be split."""
    h, w = 120, 120
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (60, 60), 30, 255, thickness=-1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = watershed_split(contours[0], mask, peak_min_distance_px=8)
    assert out == []


def test_watershed_split_side_by_side_merge():
    """Side-by-side merge — chain rule cuts wrong; watershed cuts right.

    Two disks placed close in x (parallel along their long axis); the
    chain rule's perpendicular-to-major-axis cut would slice each fly
    in half. Watershed should still separate them along the y mid-line.
    """
    h, w = 200, 240
    mask = np.zeros((h, w), dtype=np.uint8)
    # Two horizontally-elongated ellipses side-by-side along x.
    cv2.ellipse(mask, (90, 100), (28, 12), 0, 0, 360, 255, thickness=-1)
    cv2.ellipse(mask, (140, 100), (28, 12), 0, 0, 360, 255, thickness=-1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert len(contours) == 1
    out = watershed_split(contours[0], mask, peak_min_distance_px=8)
    assert len(out) == 2


def test_watershed_split_handles_zero_blob_gracefully():
    h, w = 100, 100
    mask = np.zeros((h, w), dtype=np.uint8)
    fake_contour = np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]],
                            dtype=np.int32)
    # The contour is non-empty but the rasterised blob inside this
    # all-zero mask is just a 10×10 square; should yield 1 peak → [].
    out = watershed_split(fake_contour, mask, peak_min_distance_px=8)
    assert out == []


# ─────────────────────────────────────────────────────────────────
#  WatershedSplitter
# ─────────────────────────────────────────────────────────────────


def test_splitter_passes_through_when_below_threshold():
    """A contour smaller than the threshold is returned untouched."""
    mask, contour = _two_disk_blob()
    # area_ratio_threshold × expected_animal_area_px = 1.5 × 100_000 → no split.
    splitter = WatershedSplitter(expected_animal_area_px=100_000.0)
    out = splitter.maybe_split([contour], mask)
    assert out == [contour]


def test_splitter_splits_oversized_two_disks():
    mask, contour = _two_disk_blob()
    expected = 0.5 * float(cv2.contourArea(contour))  # half the merged area
    splitter = WatershedSplitter(expected_animal_area_px=expected)
    out = splitter.maybe_split([contour], mask)
    assert len(out) == 2


def test_splitter_falls_back_to_chain_when_watershed_finds_one_peak():
    """A featureless oversized blob — watershed finds < 2 peaks → chain rule."""
    h, w = 200, 200
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (100, 100), (60, 12), 0, 0, 360, 255, thickness=-1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]
    # Oversized vs expected.
    expected = 0.4 * float(cv2.contourArea(contour))
    splitter = WatershedSplitter(
        expected_animal_area_px=expected, chain_fallback=True,
    )
    out = splitter.maybe_split([contour], mask)
    # Chain rule cuts perpendicular to major axis at centroid → 2 halves.
    assert len(out) == 2

    # Same scenario but fallback disabled → original kept.
    splitter_no_fb = WatershedSplitter(
        expected_animal_area_px=expected, chain_fallback=False,
    )
    out_no_fb = splitter_no_fb.maybe_split([contour], mask)
    assert len(out_no_fb) == 1


def test_splitter_learning_phase_returns_unchanged():
    """During the auto-learn window the splitter must not split anything."""
    mask, contour = _two_disk_blob()
    splitter = WatershedSplitter(learn_frames=5)
    assert splitter.is_learning
    out = splitter.maybe_split([contour], mask)
    assert out == [contour]


def test_splitter_learns_expected_area_after_n_frames():
    splitter = WatershedSplitter(learn_frames=3)
    fake_contour = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]],
                            dtype=np.int32)
    mask = np.zeros((20, 20), dtype=np.uint8)
    for _ in range(3):
        splitter.maybe_split([fake_contour], mask)
    assert not splitter.is_learning
    assert splitter.expected_animal_area_px == pytest.approx(100.0)


def test_splitter_requires_fg_mask_after_learning():
    """Once trained, calling maybe_split without fg_mask must raise."""
    mask, contour = _two_disk_blob()
    splitter = WatershedSplitter(expected_animal_area_px=100.0)
    with pytest.raises(ValueError, match="fg_mask"):
        splitter.maybe_split([contour])


def test_splitter_validates_constructor_args():
    with pytest.raises(ValueError):
        WatershedSplitter(area_ratio_threshold=1.0)
    with pytest.raises(ValueError):
        WatershedSplitter(learn_frames=0)
    with pytest.raises(ValueError):
        WatershedSplitter(peak_min_distance_px=0)
    with pytest.raises(ValueError):
        WatershedSplitter(peak_min_depth_px=-1.0)
    with pytest.raises(ValueError):
        WatershedSplitter(expected_animal_area_px=0.0)

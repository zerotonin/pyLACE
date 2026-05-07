"""Pose-normalised fingerprint and ToxTrac continuity scoring."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pylace.posthoc.appearance import (
    RunningMedianFingerprint,
    area_continuity_score,
    axis_ratio_continuity_score,
    extract_patch,
    is_confident_frame,
    match_patch,
)


# ─────────────────────────────────────────────────────────────────
#  extract_patch
# ─────────────────────────────────────────────────────────────────


def test_extract_patch_centres_dark_blob_on_bright_background():
    """The dark fly silhouette ends up in the patch centre."""
    frame = np.full((200, 300), 200, dtype=np.uint8)
    cv2.ellipse(frame, (150, 100), (40, 15), 0, 0, 360, 60, thickness=-1)
    patch = extract_patch(
        frame, cx=150, cy=100, orientation_deg=90.0,
        major_axis_px=80.0, minor_axis_px=30.0, patch_h=16, patch_w=32,
    )
    assert patch.shape == (16, 32)
    assert patch.dtype == np.uint8
    centre = patch[6:10, 14:18].astype(np.float32).mean()
    corners = patch[[0, 0, -1, -1], [0, -1, 0, -1]].astype(np.float32).mean()
    assert centre < 100, "fly body should be dark in the patch centre"
    assert corners > 150, "background should be bright at the corners"


def test_extract_patch_aligns_major_axis_horizontally():
    """A vertical-major-axis ellipse and a horizontal one produce similar patches."""
    frame_h = np.full((200, 200), 200, dtype=np.uint8)
    frame_v = np.full((200, 200), 200, dtype=np.uint8)
    # Horizontal major axis: ellipse axes (40, 15) at angle 0 → fitEllipse angle ~ 90.
    cv2.ellipse(frame_h, (100, 100), (40, 15), 0, 0, 360, 60, thickness=-1)
    # Vertical major axis: ellipse axes (15, 40) at angle 0 → fitEllipse angle ~ 0.
    cv2.ellipse(frame_v, (100, 100), (15, 40), 0, 0, 360, 60, thickness=-1)
    patch_h = extract_patch(
        frame_h, cx=100, cy=100, orientation_deg=90.0,
        major_axis_px=80.0, minor_axis_px=30.0, patch_h=16, patch_w=32,
    )
    patch_v = extract_patch(
        frame_v, cx=100, cy=100, orientation_deg=0.0,
        major_axis_px=80.0, minor_axis_px=30.0, patch_h=16, patch_w=32,
    )
    rmse = match_patch(patch_h, patch_v)
    assert rmse < 10.0, f"pose-normalisation should align both, got rmse={rmse}"


# ─────────────────────────────────────────────────────────────────
#  match_patch
# ─────────────────────────────────────────────────────────────────


def test_match_patch_self_match_is_zero():
    rng = np.random.default_rng(0)
    patch = rng.integers(0, 255, size=(16, 32), dtype=np.uint8)
    assert match_patch(patch, patch) == pytest.approx(0.0)


def test_match_patch_handles_180_degree_flip():
    """A patch and its 180-deg flip should match perfectly via the flip branch."""
    rng = np.random.default_rng(1)
    patch = rng.integers(0, 255, size=(16, 32), dtype=np.uint8)
    flipped = np.flip(patch, axis=(0, 1))
    assert match_patch(patch, flipped) == pytest.approx(0.0)


def test_match_patch_distinguishes_genuinely_different_patches():
    a = np.full((16, 32), 50, dtype=np.uint8)
    b = np.full((16, 32), 200, dtype=np.uint8)
    assert match_patch(a, b) == pytest.approx(150.0, abs=1e-3)


def test_match_patch_shape_mismatch_raises():
    a = np.zeros((16, 32), dtype=np.uint8)
    b = np.zeros((8, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        match_patch(a, b)


# ─────────────────────────────────────────────────────────────────
#  Continuity scores
# ─────────────────────────────────────────────────────────────────


def test_axis_ratio_continuity_score_zero_on_match():
    assert axis_ratio_continuity_score(80.0, 30.0, 80.0 / 30.0) == pytest.approx(0.0)


def test_axis_ratio_continuity_score_normalises_by_reference():
    # Candidate axis_ratio = 4.0, ref = 2.0 → diff = 2.0, normalised = 1.0.
    assert axis_ratio_continuity_score(80.0, 20.0, 2.0) == pytest.approx(1.0)


def test_area_continuity_score_zero_on_match():
    assert area_continuity_score(1000.0, 1000.0) == pytest.approx(0.0)


def test_area_continuity_score_normalises_by_reference():
    assert area_continuity_score(1500.0, 1000.0) == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────
#  Confidence detection
# ─────────────────────────────────────────────────────────────────


def test_is_confident_frame_accepts_clean_three_fly_frame():
    cx = np.array([10.0, 200.0, 50.0])
    cy = np.array([10.0, 100.0, 150.0])
    area = np.array([1000.0, 1050.0, 1020.0])
    assert is_confident_frame(
        cx, cy, area, n_animals=3, min_pairwise_distance_px=20,
        expected_area_px=1000.0, area_tol=0.3,
    )


def test_is_confident_frame_rejects_near_merge():
    cx = np.array([10.0, 12.0, 50.0])
    cy = np.array([10.0, 12.0, 150.0])
    area = np.array([1000.0, 1050.0, 1020.0])
    assert not is_confident_frame(
        cx, cy, area, n_animals=3, min_pairwise_distance_px=20,
        expected_area_px=1000.0, area_tol=0.3,
    )


def test_is_confident_frame_rejects_oversized_blob():
    cx = np.array([10.0, 200.0, 50.0])
    cy = np.array([10.0, 100.0, 150.0])
    area = np.array([1000.0, 2500.0, 1020.0])
    assert not is_confident_frame(
        cx, cy, area, n_animals=3, min_pairwise_distance_px=20,
        expected_area_px=1000.0, area_tol=0.3,
    )


def test_is_confident_frame_rejects_wrong_count():
    cx = np.array([10.0, 200.0])
    cy = np.array([10.0, 100.0])
    area = np.array([1000.0, 1050.0])
    assert not is_confident_frame(
        cx, cy, area, n_animals=3, min_pairwise_distance_px=20,
        expected_area_px=1000.0, area_tol=0.3,
    )


# ─────────────────────────────────────────────────────────────────
#  RunningMedianFingerprint
# ─────────────────────────────────────────────────────────────────


def test_running_median_returns_none_for_unknown_id():
    rmf = RunningMedianFingerprint()
    assert rmf.median_patch(99) is None
    assert rmf.median_area(99) is None
    assert rmf.median_axis_ratio(99) is None


def test_running_median_caps_history_at_history_frames():
    rmf = RunningMedianFingerprint(history_frames=5)
    patch = np.full((4, 8), 100, dtype=np.uint8)
    for area in range(20):
        rmf.update(0, patch=patch, area_px=float(area), major_axis_px=20.0,
                   minor_axis_px=10.0)
    # Only the last 5 areas (15..19) should remain → median = 17.0.
    assert rmf.median_area(0) == pytest.approx(17.0)


def test_running_median_patch_returns_pixelwise_median():
    rmf = RunningMedianFingerprint(history_frames=5)
    a = np.full((4, 8), 50, dtype=np.uint8)
    b = np.full((4, 8), 200, dtype=np.uint8)
    rmf.update(0, patch=a, area_px=1000.0, major_axis_px=20.0, minor_axis_px=10.0)
    rmf.update(0, patch=a, area_px=1000.0, major_axis_px=20.0, minor_axis_px=10.0)
    rmf.update(0, patch=b, area_px=1000.0, major_axis_px=20.0, minor_axis_px=10.0)
    med = rmf.median_patch(0)
    # Median of [50, 50, 200] = 50.
    assert med is not None
    assert (med == 50).all()


def test_running_median_validates_history_frames():
    with pytest.raises(ValueError, match="history_frames"):
        RunningMedianFingerprint(history_frames=0)

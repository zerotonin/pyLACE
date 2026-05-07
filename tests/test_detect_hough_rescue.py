"""LACE-paper Hough rescue: ellipse candidates for under-counted frames."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pylace.detect.frame import Detection
from pylace.detect.hough_rescue import (
    HoughRescuer,
    generate_candidates,
)


def _two_overlapping_ellipses(w: int = 200, h: int = 200) -> np.ndarray:
    """Two ellipses fully overlapping into a single contour with no neck."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (95, 100), (28, 12), 0, 0, 360, 255, thickness=-1)
    cv2.ellipse(mask, (105, 100), (28, 12), 0, 0, 360, 255, thickness=-1)
    return mask


def _det(cx: float, cy: float, area: float = 1100.0) -> Detection:
    return Detection(
        cx=cx, cy=cy, area_px=area, perimeter_px=120.0, solidity=0.95,
        major_axis_px=56.0, minor_axis_px=24.0, orientation_deg=0.0,
        contour=None, track_id=-1,
    )


# ─────────────────────────────────────────────────────────────────
#  Pure helper: generate_candidates
# ─────────────────────────────────────────────────────────────────


def test_generate_candidates_finds_ellipse_in_a_round_blob():
    """A single elliptical blob → at least one candidate near its area."""
    h, w = 200, 200
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (100, 100), (28, 12), 0, 0, 360, 255, thickness=-1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    expected_area = float(np.pi * 28 * 12)  # ≈ 1056
    cands = generate_candidates(
        contours, expected_area=expected_area, area_tolerance=0.5,
    )
    assert len(cands) >= 1
    # The whole-contour fit should be near the centre.
    assert any(abs(c.cx - 100) < 10 and abs(c.cy - 100) < 10 for c in cands)


def test_generate_candidates_filters_by_area_tolerance():
    """A blob far from expected area produces zero candidates within tolerance."""
    h, w = 200, 200
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (100, 100), (60, 50), 0, 0, 360, 255, thickness=-1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cands = generate_candidates(
        contours, expected_area=300.0, area_tolerance=0.3,
    )
    # A 60×50 ellipse has area ~9425; expected 300 ± 30% = [210, 390].
    # No candidate fits.
    assert cands == []


# ─────────────────────────────────────────────────────────────────
#  HoughRescuer
# ─────────────────────────────────────────────────────────────────


def test_rescuer_passes_through_when_count_meets_target():
    """3 detections in a 3-fly arena → no rescue needed."""
    rescuer = HoughRescuer(target_n=3, expected_animal_area_px=1056.0)
    mask = _two_overlapping_ellipses()
    dets = [_det(95, 100), _det(105, 100), _det(160, 100)]
    out = rescuer.maybe_rescue(dets, mask)
    assert out == dets


def test_rescuer_rescues_a_two_fly_overlap_to_target_n():
    """Two flies in an unsplittable merge → rescue adds the missing one."""
    h, w = 200, 200
    # 1 merged blob + 1 separate fly → only 2 detections.
    mask = _two_overlapping_ellipses(w, h)
    cv2.ellipse(mask, (160, 100), (28, 12), 0, 0, 360, 255, thickness=-1)
    # Pretend the existing pipeline only returned the right-hand fly +
    # the merged blob fitted as a single big detection (area off but
    # centroid roughly mid-merge).
    existing = [_det(100, 100, area=2200.0), _det(160, 100, area=1100.0)]
    rescuer = HoughRescuer(target_n=3, expected_animal_area_px=1100.0,
                           area_tolerance=0.6)
    out = rescuer.maybe_rescue(existing, mask)
    assert len(out) == 3
    # The new candidate should sit somewhere on the merged blob and
    # not within NMS distance of the existing centres.
    new = out[2]
    body_radius = float(np.sqrt(1100.0 / np.pi))
    for d in existing:
        assert np.hypot(new.cx - d.cx, new.cy - d.cy) >= body_radius


def test_rescuer_caps_at_target_n_minus_existing():
    """Even with many candidates, we add at most ``target_n - existing``."""
    h, w = 200, 200
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (50, 50), (28, 12), 0, 0, 360, 255, thickness=-1)
    cv2.ellipse(mask, (150, 50), (28, 12), 0, 0, 360, 255, thickness=-1)
    cv2.ellipse(mask, (50, 150), (28, 12), 0, 0, 360, 255, thickness=-1)
    cv2.ellipse(mask, (150, 150), (28, 12), 0, 0, 360, 255, thickness=-1)
    existing = [_det(50, 50)]
    rescuer = HoughRescuer(target_n=2, expected_animal_area_px=1056.0)
    out = rescuer.maybe_rescue(existing, mask)
    assert len(out) == 2
    # Newly added must not overlap the existing one.
    new = out[1]
    body_radius = float(np.sqrt(1056.0 / np.pi))
    assert np.hypot(new.cx - 50, new.cy - 50) >= body_radius


def test_rescuer_learning_phase_returns_unchanged():
    """During the auto-learn window the rescuer is a pass-through."""
    rescuer = HoughRescuer(target_n=3, learn_frames=5)
    assert rescuer.is_learning
    dets = [_det(50, 50)]
    out = rescuer.maybe_rescue(dets, np.zeros((100, 100), dtype=np.uint8))
    assert out == dets
    assert rescuer.is_learning


def test_rescuer_learns_expected_area_after_n_frames():
    rescuer = HoughRescuer(target_n=3, learn_frames=3)
    for _ in range(3):
        rescuer.maybe_rescue([_det(0, 0, area=2500.0)],
                              np.zeros((10, 10), dtype=np.uint8))
    assert not rescuer.is_learning
    assert rescuer.expected_animal_area_px == pytest.approx(2500.0)


def test_rescuer_validates_constructor_args():
    with pytest.raises(ValueError):
        HoughRescuer(target_n=0)
    with pytest.raises(ValueError):
        HoughRescuer(target_n=2, window_min_points=4)
    with pytest.raises(ValueError):
        HoughRescuer(target_n=2, window_fraction=0.0)
    with pytest.raises(ValueError):
        HoughRescuer(target_n=2, area_tolerance=0.0)
    with pytest.raises(ValueError):
        HoughRescuer(target_n=2, area_tolerance=1.0)
    with pytest.raises(ValueError):
        HoughRescuer(target_n=2, learn_frames=0)
    with pytest.raises(ValueError):
        HoughRescuer(target_n=2, expected_animal_area_px=0.0)

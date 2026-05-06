# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — tests.test_widgets_navigation                          ║
# ║  « pure-logic helpers for the three-bar navigation widget »      ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Tests for :class:`pylace.widgets.navigation.FrameRange`."""

from __future__ import annotations

import pytest

from pylace.widgets.navigation import FrameRange


def test_default_range_spans_full_movie():
    fr = FrameRange(total_frames=1000)
    assert fr.lo == 0
    assert fr.hi == 999


def test_explicit_range_is_clamped_to_movie():
    fr = FrameRange(total_frames=1000, lo=-50, hi=2_000)
    assert fr.lo == 0
    assert fr.hi == 999


def test_lo_must_remain_below_hi():
    fr = FrameRange(total_frames=1000, lo=500, hi=400)
    # __post_init__ pushes hi above lo so the bottom slider has a real range.
    assert fr.hi > fr.lo


def test_fraction_at_extremes():
    fr = FrameRange(total_frames=1000)
    assert fr.fraction(0) == 0.0
    assert fr.fraction(999) == 1.0
    assert 0.49 < fr.fraction(500) < 0.51


def test_fraction_clips_out_of_range():
    fr = FrameRange(total_frames=1000)
    assert fr.fraction(-100) == 0.0
    assert fr.fraction(2_000) == 1.0


def test_clamp_to_zoom_window():
    fr = FrameRange(total_frames=1000, lo=200, hi=300)
    assert fr.clamp(150) == 200
    assert fr.clamp(250) == 250
    assert fr.clamp(400) == 300


def test_zero_total_frames_raises():
    with pytest.raises(ValueError):
        FrameRange(total_frames=0)

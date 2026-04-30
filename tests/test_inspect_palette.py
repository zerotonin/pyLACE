# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — tests.test_inspect_palette                             ║
# ║  « per-track BGR colour generator »                              ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Tests for :func:`pylace.inspect.palette.palette_bgr`."""

from __future__ import annotations

import pytest

from pylace.inspect.palette import palette_bgr


def test_palette_zero_is_empty():
    assert palette_bgr(0) == []


def test_palette_negative_raises():
    with pytest.raises(ValueError):
        palette_bgr(-1)


def test_palette_3_uses_first_three_wong_colours():
    """The first three are always Wong vermilion / sky_blue / bluish_green."""
    p = palette_bgr(3)
    assert len(p) == 3
    # Wong vermilion #D55E00 → BGR (0, 94, 213).
    assert p[0] == (0, 94, 213)


def test_palette_8_returns_eight_distinct_colours():
    p = palette_bgr(8)
    assert len(p) == 8
    assert len(set(p)) == 8


def test_palette_12_extends_with_hsv_samples():
    p = palette_bgr(12)
    assert len(p) == 12
    # All 12 are distinct.
    assert len(set(p)) == 12
    # First 8 are unchanged from the 8-colour set.
    assert p[:8] == palette_bgr(8)


def test_palette_returns_three_byte_tuples():
    p = palette_bgr(20)
    for r in p:
        assert isinstance(r, tuple)
        assert len(r) == 3
        assert all(0 <= c <= 255 for c in r)

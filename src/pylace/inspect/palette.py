# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — inspect.palette                                        ║
# ║  « per-track BGR colours, Wong → golden-angle HSV »              ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Generate distinct per-track BGR colours."""

from __future__ import annotations

import colorsys

from pylace.roi.constants import WONG, _hex_to_bgr

_WONG_ORDER = (
    "vermilion",
    "sky_blue",
    "bluish_green",
    "orange",
    "yellow",
    "reddish_purple",
    "blue",
    "black",
)


def palette_bgr(n: int) -> list[tuple[int, int, int]]:
    """Return ``n`` distinct BGR colours.

    The first 8 come from the Wong (2011) colourblind-safe palette in a
    visually-ordered sequence; for n > 8, additional colours are sampled
    on the HSV circle at the golden-angle stride so they stay maximally
    distinct from each other and from the Wong colours.
    """
    if n < 0:
        raise ValueError("n must be >= 0.")
    out: list[tuple[int, int, int]] = []
    for i in range(min(n, len(_WONG_ORDER))):
        out.append(_hex_to_bgr(WONG[_WONG_ORDER[i]]))
    if n > len(_WONG_ORDER):
        for i in range(n - len(_WONG_ORDER)):
            hue = ((i + len(_WONG_ORDER)) * 0.61803398875) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            out.append((int(b * 255), int(g * 255), int(r * 255)))
    return out


__all__ = ["palette_bgr"]

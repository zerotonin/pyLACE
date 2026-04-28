"""Boolean mask matching the arena boundary — True inside, False outside."""

from __future__ import annotations

import cv2
import numpy as np

from pylace.annotator.geometry import Arena, Circle


def arena_mask(arena: Arena, frame_size: tuple[int, int]) -> np.ndarray:
    """Boolean mask, True for pixels inside the arena.

    Args:
        arena: Circle / Rectangle / Polygon.
        frame_size: ``(width, height)`` of the destination frame, in pixels.

    Returns:
        Boolean ``(height, width)`` array.
    """
    width, height = frame_size
    if isinstance(arena, Circle):
        return _circle_mask(arena, width, height)
    return _polygon_mask(arena.vertices, width, height)


def _circle_mask(circle: Circle, width: int, height: int) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return (xx - circle.cx) ** 2 + (yy - circle.cy) ** 2 <= circle.r ** 2


def _polygon_mask(
    vertices: list[tuple[float, float]], width: int, height: int,
) -> np.ndarray:
    poly = np.array(vertices, dtype=np.int32).reshape(-1, 1, 2)
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(canvas, [poly], 255)
    return canvas.astype(bool)

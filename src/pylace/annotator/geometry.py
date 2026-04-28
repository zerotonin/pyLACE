"""Arena geometry, world-frame, and pix-to-mm calibration types."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Vertex = tuple[float, float]


@dataclass
class Circle:
    """Circular arena boundary in image (pixel) coordinates."""

    cx: float
    cy: float
    r: float

    def origin_candidates(self) -> list[tuple[str, Vertex]]:
        """Real points on the shape that the user can snap the world origin to.

        Returns:
            (label, (x, y)) pairs for east, north, west, south rim points and
            the centre, in image coordinates.
        """
        return [
            ("east", (self.cx + self.r, self.cy)),
            ("north", (self.cx, self.cy - self.r)),
            ("west", (self.cx - self.r, self.cy)),
            ("south", (self.cx, self.cy + self.r)),
            ("centre", (self.cx, self.cy)),
        ]


@dataclass
class Rectangle:
    """Axis-aligned rectangular arena, stored as four corner vertices."""

    vertices: list[Vertex]

    def __post_init__(self) -> None:
        if len(self.vertices) != 4:
            raise ValueError("Rectangle requires exactly 4 vertices.")

    def origin_candidates(self) -> list[tuple[str, Vertex]]:
        return [(f"corner_{i}", v) for i, v in enumerate(self.vertices)]

    @classmethod
    def from_two_points(cls, p1: Vertex, p2: Vertex) -> Rectangle:
        """Axis-aligned rectangle from two opposite corners (any diagonal)."""
        x1, y1 = p1
        x2, y2 = p2
        x_min, x_max = sorted((x1, x2))
        y_min, y_max = sorted((y1, y2))
        return cls(
            [
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
            ]
        )


@dataclass
class Polygon:
    """Free-form polygonal arena."""

    vertices: list[Vertex]

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError("Polygon requires at least 3 vertices.")

    def origin_candidates(self) -> list[tuple[str, Vertex]]:
        return [(f"vertex_{i}", v) for i, v in enumerate(self.vertices)]


Arena = Circle | Rectangle | Polygon


def shape_name(arena: Arena) -> str:
    """JSON discriminator string for an arena instance."""
    if isinstance(arena, Circle):
        return "circle"
    if isinstance(arena, Rectangle):
        return "rectangle"
    if isinstance(arena, Polygon):
        return "polygon"
    raise TypeError(f"Unknown arena type: {type(arena).__name__}")


@dataclass
class WorldFrame:
    """Cartesian world frame anchored to a real point on the arena."""

    origin_pixel: Vertex
    y_axis: Literal["up", "down"] = "up"
    x_axis: Literal["right", "left"] = "right"


@dataclass
class Calibration:
    """Pix-to-mm calibration derived from a known physical dimension."""

    reference_kind: Literal["diameter", "edge"]
    physical_mm: float
    pixel_distance: float
    reference_vertices: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.physical_mm <= 0:
            raise ValueError("physical_mm must be positive.")
        if self.pixel_distance <= 0:
            raise ValueError("pixel_distance must be positive.")
        if self.reference_kind == "edge" and self.reference_vertices is None:
            raise ValueError("Edge calibration requires reference_vertices.")
        if self.reference_kind == "diameter" and self.reference_vertices is not None:
            raise ValueError("Diameter calibration must not set reference_vertices.")

    @property
    def mm_per_pixel(self) -> float:
        return self.physical_mm / self.pixel_distance


def edge_length(vertices: list[Vertex], i: int, j: int) -> float:
    """Euclidean distance between two vertices in pixel space."""
    ax, ay = vertices[i]
    bx, by = vertices[j]
    return math.hypot(bx - ax, by - ay)


def pixel_to_world(point: Vertex, frame: WorldFrame, cal: Calibration) -> Vertex:
    """Convert an image-pixel point to arena-local mm coordinates."""
    px, py = point
    ox, oy = frame.origin_pixel
    mm_x = (px - ox) * cal.mm_per_pixel
    mm_y = (py - oy) * cal.mm_per_pixel
    if frame.x_axis == "left":
        mm_x = -mm_x
    if frame.y_axis == "up":
        # Image coords are y-down; flip to make a "y up" world frame.
        mm_y = -mm_y
    return (mm_x, mm_y)

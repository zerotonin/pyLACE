"""Shared pytest fixtures for the pyLACE test suite."""

import numpy as np
import pytest


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """A 32x32 grayscale frame with a single dark elliptical blob.

    Tiny enough that detection unit tests run in milliseconds.
    """
    h = w = 32
    frame = np.full((h, w), 255, dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    cy, cx = 16.0, 16.0
    a, b = 8.0, 3.0
    mask = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1.0
    frame[mask] = 40
    return frame

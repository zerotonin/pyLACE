"""Phase 3 — multi-fly metrics: pairwise distance, NN distance, polarisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pylace.posthoc.multifly import (
    nearest_neighbour_distance,
    pairwise_distance_matrix,
    polarisation,
)


def _multi_track_df(
    tracks: dict[int, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Build a multi-track cleaned-trajectory frame from per-track arrays."""
    rows = []
    for tid, data in tracks.items():
        n = data["cx"].size
        for i in range(n):
            rows.append({
                "frame_idx": int(data["frame"][i]),
                "track_id": int(tid),
                "cx_smooth_px": float(data["cx"][i]),
                "cy_smooth_px": float(data["cy"][i]),
                "heading_deg": (
                    float(data["heading"][i])
                    if "heading" in data else float("nan")
                ),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
#  Pairwise distance matrix
# ─────────────────────────────────────────────────────────────────


def test_pairwise_distance_matrix_two_tracks_constant_distance():
    n = 10
    fr = np.arange(n)
    df = _multi_track_df({
        0: {"frame": fr, "cx": np.full(n, 0.0), "cy": np.full(n, 0.0)},
        1: {"frame": fr, "cx": np.full(n, 30.0), "cy": np.full(n, 40.0)},
    })
    d, frames, ids = pairwise_distance_matrix(df, pix_per_mm=10.0)
    assert ids == [0, 1]
    assert d.shape == (n, 2, 2)
    assert np.allclose(np.diagonal(d, axis1=1, axis2=2), 0.0)
    # 30² + 40² = 50 px → 5 mm.
    assert np.allclose(d[:, 0, 1], 5.0)
    assert np.allclose(d[:, 1, 0], 5.0)


def test_pairwise_distance_matrix_propagates_nan():
    fr = np.arange(5)
    df = _multi_track_df({
        0: {"frame": fr, "cx": np.array([0, np.nan, 0, 0, 0], float),
                          "cy": np.zeros(5)},
        1: {"frame": fr, "cx": np.full(5, 10.0), "cy": np.zeros(5)},
    })
    d, _, _ = pairwise_distance_matrix(df, pix_per_mm=1.0)
    assert np.isnan(d[1, 0, 1])
    assert d[0, 0, 1] == pytest.approx(10.0)


def test_pairwise_distance_matrix_rejects_non_positive_pix_per_mm():
    df = _multi_track_df({
        0: {"frame": np.arange(2), "cx": np.zeros(2), "cy": np.zeros(2)},
        1: {"frame": np.arange(2), "cx": np.ones(2), "cy": np.zeros(2)},
    })
    with pytest.raises(ValueError, match="pix_per_mm"):
        pairwise_distance_matrix(df, pix_per_mm=0.0)


# ─────────────────────────────────────────────────────────────────
#  Nearest neighbour distance
# ─────────────────────────────────────────────────────────────────


def test_nn_distance_three_tracks_picks_the_minimum_other():
    n = 4
    fr = np.arange(n)
    df = _multi_track_df({
        0: {"frame": fr, "cx": np.full(n, 0.0),  "cy": np.zeros(n)},
        1: {"frame": fr, "cx": np.full(n, 10.0), "cy": np.zeros(n)},
        2: {"frame": fr, "cx": np.full(n, 50.0), "cy": np.zeros(n)},
    })
    out = nearest_neighbour_distance(df, pix_per_mm=10.0)
    by = out.set_index(["frame_idx", "track_id"])["nn_distance_mm"]
    # track 0 closest to track 1, distance 10 px → 1 mm.
    assert by[0, 0] == pytest.approx(1.0)
    assert by[0, 1] == pytest.approx(1.0)
    # track 2 closest to track 1, distance 40 px → 4 mm.
    assert by[0, 2] == pytest.approx(4.0)


def test_nn_distance_returns_empty_for_single_track():
    df = _multi_track_df({
        0: {"frame": np.arange(3), "cx": np.zeros(3), "cy": np.zeros(3)},
    })
    out = nearest_neighbour_distance(df, pix_per_mm=10.0)
    assert out.empty
    assert list(out.columns) == ["frame_idx", "track_id", "nn_distance_mm"]


def test_nn_distance_nan_when_no_partner_present():
    fr = np.arange(3)
    df = _multi_track_df({
        0: {"frame": fr, "cx": np.zeros(3), "cy": np.zeros(3)},
        1: {"frame": fr, "cx": np.array([10.0, np.nan, 10.0]),
                          "cy": np.zeros(3)},
    })
    out = nearest_neighbour_distance(df, pix_per_mm=10.0)
    by = out.set_index(["frame_idx", "track_id"])["nn_distance_mm"]
    assert np.isnan(by[1, 0])
    assert np.isnan(by[1, 1])
    assert by[0, 0] == pytest.approx(1.0)
    assert by[2, 0] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────
#  Polarisation
# ─────────────────────────────────────────────────────────────────


def test_polarisation_perfect_alignment_is_one():
    fr = np.arange(5)
    df = _multi_track_df({
        0: {"frame": fr, "cx": np.zeros(5), "cy": np.zeros(5),
            "heading": np.full(5, 30.0)},
        1: {"frame": fr, "cx": np.zeros(5), "cy": np.zeros(5),
            "heading": np.full(5, 30.0)},
        2: {"frame": fr, "cx": np.zeros(5), "cy": np.zeros(5),
            "heading": np.full(5, 30.0)},
    })
    out = polarisation(df)
    assert np.allclose(out["polarisation"].to_numpy(), 1.0)


def test_polarisation_three_120deg_apart_is_zero():
    fr = np.arange(2)
    df = _multi_track_df({
        0: {"frame": fr, "cx": np.zeros(2), "cy": np.zeros(2),
            "heading": np.full(2, 0.0)},
        1: {"frame": fr, "cx": np.zeros(2), "cy": np.zeros(2),
            "heading": np.full(2, 120.0)},
        2: {"frame": fr, "cx": np.zeros(2), "cy": np.zeros(2),
            "heading": np.full(2, 240.0)},
    })
    out = polarisation(df)
    assert np.allclose(out["polarisation"].to_numpy(), 0.0, atol=1e-9)


def test_polarisation_two_antiparallel_is_zero():
    fr = np.arange(2)
    df = _multi_track_df({
        0: {"frame": fr, "cx": np.zeros(2), "cy": np.zeros(2),
            "heading": np.full(2, 0.0)},
        1: {"frame": fr, "cx": np.zeros(2), "cy": np.zeros(2),
            "heading": np.full(2, 180.0)},
    })
    out = polarisation(df)
    assert np.allclose(out["polarisation"].to_numpy(), 0.0, atol=1e-9)


def test_polarisation_nan_with_one_track_only_per_frame():
    fr = np.arange(3)
    df = _multi_track_df({
        0: {"frame": fr, "cx": np.zeros(3), "cy": np.zeros(3),
            "heading": np.array([30.0, np.nan, 30.0])},
        1: {"frame": fr, "cx": np.zeros(3), "cy": np.zeros(3),
            "heading": np.array([np.nan, np.nan, 30.0])},
    })
    out = polarisation(df).set_index("frame_idx")
    assert np.isnan(out.loc[0, "polarisation"])  # only one valid
    assert np.isnan(out.loc[1, "polarisation"])  # zero valid
    assert out.loc[2, "polarisation"] == pytest.approx(1.0)


def test_polarisation_n_valid_tracks_reported():
    fr = np.arange(3)
    df = _multi_track_df({
        0: {"frame": fr, "cx": np.zeros(3), "cy": np.zeros(3),
            "heading": np.full(3, 0.0)},
        1: {"frame": fr, "cx": np.zeros(3), "cy": np.zeros(3),
            "heading": np.array([0.0, np.nan, 0.0])},
    })
    out = polarisation(df).set_index("frame_idx")
    assert int(out.loc[0, "n_valid_tracks"]) == 2
    assert int(out.loc[1, "n_valid_tracks"]) == 1
    assert int(out.loc[2, "n_valid_tracks"]) == 2

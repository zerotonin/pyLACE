"""Pure-function tests for the 4-state CV Kalman primitives."""

from __future__ import annotations

import numpy as np
import pytest

from pylace.tracking.kalman import (
    H,
    initial_covariance,
    mahalanobis_sq,
    measurement_noise,
    predict,
    process_noise,
    transition_matrix,
    update,
)


# ─────────────────────────────────────────────────────────────────
#  Matrix builders
# ─────────────────────────────────────────────────────────────────


def test_transition_matrix_advances_position_by_velocity():
    F = transition_matrix(dt=2.0)
    state = np.array([10.0, 20.0, 1.0, -1.0])
    new = F @ state
    assert np.allclose(new, [12.0, 18.0, 1.0, -1.0])


def test_transition_matrix_at_dt_one_is_identity_plus_velocity():
    F = transition_matrix(dt=1.0)
    state = np.array([0.0, 0.0, 5.0, -3.0])
    new = F @ state
    assert np.allclose(new, [5.0, -3.0, 5.0, -3.0])


def test_process_noise_is_diagonal_with_squared_stds():
    Q = process_noise(q_pos=0.5, q_vel=2.0)
    assert np.allclose(np.diag(Q), [0.25, 0.25, 4.0, 4.0])
    Q_off = Q - np.diag(np.diag(Q))
    assert np.allclose(Q_off, 0.0)


def test_process_noise_rejects_negative_inputs():
    with pytest.raises(ValueError):
        process_noise(q_pos=-1.0, q_vel=0.5)
    with pytest.raises(ValueError):
        process_noise(q_pos=0.5, q_vel=-1.0)


def test_measurement_noise_is_2x2_diagonal():
    R = measurement_noise(r_pos=2.0)
    assert R.shape == (2, 2)
    assert np.allclose(R, np.eye(2) * 4.0)


def test_initial_covariance_has_tight_pos_loose_vel():
    cov = initial_covariance(r_pos=1.0, initial_v_std=10.0)
    assert cov.shape == (4, 4)
    assert cov[0, 0] == pytest.approx(1.0)
    assert cov[1, 1] == pytest.approx(1.0)
    assert cov[2, 2] == pytest.approx(100.0)
    assert cov[3, 3] == pytest.approx(100.0)


# ─────────────────────────────────────────────────────────────────
#  Predict / update
# ─────────────────────────────────────────────────────────────────


def test_predict_advances_state_through_F():
    F = transition_matrix(dt=1.0)
    Q = process_noise(0.0, 0.0)
    state = np.array([5.0, 5.0, 2.0, 0.0])
    cov = np.eye(4)
    new_state, _ = predict(state, cov, F, Q)
    assert np.allclose(new_state, [7.0, 5.0, 2.0, 0.0])


def test_predict_inflates_covariance_by_Q():
    F = transition_matrix(dt=1.0)
    Q = process_noise(1.0, 1.0)
    cov = np.eye(4) * 0.1
    _, new_cov = predict(np.zeros(4), cov, F, Q)
    # Diagonal should grow by at least Q's diagonal (and the F·cov·Fᵀ
    # term adds further to the position diagonals).
    assert new_cov[0, 0] >= cov[0, 0] + Q[0, 0]
    assert new_cov[2, 2] == pytest.approx(cov[2, 2] + Q[2, 2])


def test_update_pulls_state_toward_measurement():
    F = transition_matrix()
    Q = process_noise(0.5, 0.5)
    R = measurement_noise(1.0)
    cov = initial_covariance(1.0, 10.0)
    state = np.array([0.0, 0.0, 0.0, 0.0])
    measurement = np.array([10.0, 10.0])
    new_state, new_cov, mahal_sq = update(state, cov, measurement, R)
    # New position should sit between (0, 0) and the measurement,
    # closer to the measurement because pos uncertainty dominates R
    # ratio? Actually with R = init pos cov, new state ≈ (5, 5).
    assert 0.0 < new_state[0] < 10.0
    assert 0.0 < new_state[1] < 10.0
    # Position uncertainty must shrink after the observation.
    assert new_cov[0, 0] < cov[0, 0]
    assert new_cov[1, 1] < cov[1, 1]
    # Mahalanobis squared is non-negative and finite.
    assert mahal_sq >= 0.0
    assert np.isfinite(mahal_sq)


def test_update_at_zero_measurement_innovation_yields_zero_mahalanobis():
    R = measurement_noise(1.0)
    cov = initial_covariance(1.0, 10.0)
    state = np.array([5.0, 5.0, 0.0, 0.0])
    new_state, _, mahal_sq = update(state, cov, np.array([5.0, 5.0]), R)
    assert mahal_sq == pytest.approx(0.0)
    # State stays at (5, 5).
    assert np.allclose(new_state[:2], [5.0, 5.0])


# ─────────────────────────────────────────────────────────────────
#  Mahalanobis
# ─────────────────────────────────────────────────────────────────


def test_mahalanobis_sq_grows_with_distance():
    R = measurement_noise(1.0)
    cov = initial_covariance(1.0, 10.0)
    state = np.array([0.0, 0.0, 0.0, 0.0])
    near = mahalanobis_sq(state, cov, np.array([1.0, 0.0]), R)
    far = mahalanobis_sq(state, cov, np.array([5.0, 0.0]), R)
    farther = mahalanobis_sq(state, cov, np.array([10.0, 0.0]), R)
    assert near < far < farther


def test_mahalanobis_sq_smaller_when_uncertainty_is_large():
    """Higher position uncertainty → easier to explain a given offset."""
    R = measurement_noise(1.0)
    cov_tight = np.diag([1.0, 1.0, 100.0, 100.0])
    cov_loose = np.diag([100.0, 100.0, 100.0, 100.0])
    state = np.zeros(4)
    measurement = np.array([5.0, 0.0])
    tight = mahalanobis_sq(state, cov_tight, measurement, R)
    loose = mahalanobis_sq(state, cov_loose, measurement, R)
    assert loose < tight


# ─────────────────────────────────────────────────────────────────
#  H matrix
# ─────────────────────────────────────────────────────────────────


def test_H_extracts_position_only():
    state = np.array([3.0, 4.0, 99.0, -99.0])
    measurement = H @ state
    assert np.allclose(measurement, [3.0, 4.0])

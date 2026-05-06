# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACE — tracking.kalman                                        ║
# ║  « 4-state constant-velocity Kalman filter primitives »          ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Pure-numpy filter math used by the per-track motion model in   ║
# ║  ``pylace.tracking.tracks.Tracker``. State vector per track:    ║
# ║                                                                  ║
# ║      x = [px, py, vx, vy]ᵀ                                       ║
# ║                                                                  ║
# ║  Constant-velocity transition F (with dt = 1 frame), position-   ║
# ║  only measurement matrix H, diagonal process noise Q split into  ║
# ║  position and velocity components, and a 2×2 measurement noise   ║
# ║  R. Predict / update / Mahalanobis distance are exposed as       ║
# ║  small pure functions so the Tracker stays free of filter math   ║
# ║  and the unit tests can exercise the filter directly.            ║
# ║                                                                  ║
# ║  Bewley et al. SORT (2016) uses exactly this 4-state model;      ║
# ║  TRex (Walter & Couzin, eLife 2021) and idtracker.ai's           ║
# ║  deterministic post-processing both rely on it for short-term    ║
# ║  prediction during occlusion. The diagonal Q is a tunability /   ║
# ║  numerical-stability simplification of the full continuous-      ║
# ║  Wiener Q; it parametrises position-drift and velocity-jitter    ║
# ║  separately, which is easier to reason about than the off-       ║
# ║  diagonal version.                                               ║
# ╚══════════════════════════════════════════════════════════════════╝
"""4-state constant-velocity Kalman filter primitives."""

from __future__ import annotations

import numpy as np

# Position-only measurement.
H = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0]],
    dtype=np.float64,
)


def transition_matrix(dt: float = 1.0) -> np.ndarray:
    """Constant-velocity 4×4 transition: ``x' = x + vx·dt``, ``v' = v``."""
    F = np.eye(4, dtype=np.float64)
    F[0, 2] = dt
    F[1, 3] = dt
    return F


def process_noise(q_pos: float, q_vel: float) -> np.ndarray:
    """Diagonal Q with separate position-drift and velocity-jitter stds.

    Args:
        q_pos: per-frame position-drift std (px). Covers small irregular
            motions the constant-velocity model does not predict.
        q_vel: per-frame velocity-jitter std (px/frame). Covers
            saccadic acceleration. Roughly the rms acceleration the
            filter expects to see between frames.
    """
    if q_pos < 0 or q_vel < 0:
        raise ValueError(
            f"q_pos and q_vel must be non-negative; got q_pos={q_pos}, q_vel={q_vel}",
        )
    return np.diag([q_pos**2, q_pos**2, q_vel**2, q_vel**2])


def measurement_noise(r_pos: float) -> np.ndarray:
    """2×2 measurement noise; ``r_pos`` is per-axis pixel std."""
    if r_pos <= 0:
        raise ValueError(f"r_pos must be positive; got {r_pos}")
    return np.eye(2, dtype=np.float64) * (r_pos ** 2)


def initial_covariance(r_pos: float, initial_v_std: float) -> np.ndarray:
    """Birth covariance: tight position from observation, generous velocity.

    A new track has a measured position (uncertainty ≈ R) but an
    unknown velocity. ``initial_v_std`` puts a Gaussian prior on
    velocity in each axis; pick it generously enough to cover the
    fastest plausible per-frame motion (≈ saccade peak).
    """
    if r_pos <= 0 or initial_v_std <= 0:
        raise ValueError(
            f"r_pos and initial_v_std must be positive; "
            f"got r_pos={r_pos}, initial_v_std={initial_v_std}",
        )
    return np.diag([r_pos**2, r_pos**2, initial_v_std**2, initial_v_std**2])


def predict(
    state: np.ndarray, cov: np.ndarray, F: np.ndarray, Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One time step of the Kalman predict equations."""
    new_state = F @ state
    new_cov = F @ cov @ F.T + Q
    return new_state, new_cov


def update(
    state: np.ndarray, cov: np.ndarray,
    measurement: np.ndarray, R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Standard Kalman update; returns ``(state, cov, mahalanobis_sq)``.

    The squared Mahalanobis distance is returned alongside because
    every update naturally produces it (the innovation covariance is
    inverted to compute the Kalman gain). Callers that only want the
    distance use :func:`mahalanobis_sq`.
    """
    y = measurement - H @ state
    S = H @ cov @ H.T + R
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return state, cov, float("inf")
    K = cov @ H.T @ S_inv
    new_state = state + K @ y
    I = np.eye(state.size)
    new_cov = (I - K @ H) @ cov
    mahal_sq = float(y @ S_inv @ y)
    return new_state, new_cov, mahal_sq


def mahalanobis_sq(
    state: np.ndarray, cov: np.ndarray,
    measurement: np.ndarray, R: np.ndarray,
) -> float:
    """Squared Mahalanobis distance from the predicted measurement."""
    y = measurement - H @ state
    S = H @ cov @ H.T + R
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(y @ S_inv @ y)


def predicted_position(state: np.ndarray) -> tuple[float, float]:
    return float(state[0]), float(state[1])


def predicted_velocity(state: np.ndarray) -> tuple[float, float]:
    return float(state[2]), float(state[3])


__all__ = [
    "H",
    "initial_covariance",
    "mahalanobis_sq",
    "measurement_noise",
    "predict",
    "predicted_position",
    "predicted_velocity",
    "process_noise",
    "transition_matrix",
    "update",
]

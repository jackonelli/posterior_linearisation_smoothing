"""Analytical cost functions for the extended Kalman filter/RTS smoother"""
import logging
import numpy as np
from src.models.base import MotionModel, MeasModel
from src.slr.base import Slr

LOGGER = logging.getLogger(__name__)


def analytical_smoothing_cost(traj, measurements, m_1_0, P_1_0, motion_model: MotionModel, meas_model: MeasModel):
    """Cost function for an optimisation problem used in the family of extended smoothers
    Efficient implementation which assumes that the motion and meas models have no explicit dependency on the time step.

    GN optimisation of this cost function will result in a linearised function
    corresponding to the Iterated Extended Kalman Smoother (IEKS).

    Args:
        traj: states for a time sequence 1, ..., K
            represented as a np.array(K, D_x).
            (The actual variable in the cost function)
        measurements: measurements for a time sequence 1, ..., K
            represented as a list of length K of np.array(D_y,)
    """
    K = len(measurements)
    prior_diff = traj[0, :] - m_1_0
    _cost = prior_diff.T @ np.linalg.inv(P_1_0) @ prior_diff

    motion_diff = traj[1:, :] - motion_model.map_set(traj[:-1, :], None)
    meas_diff = measurements - meas_model.map_set(traj, None)
    for k in range(0, K - 1):
        _cost += motion_diff[k, :].T @ np.linalg.inv(motion_model.proc_noise(k)) @ motion_diff[k, :]
        # measurements are zero indexed, i.e. k-1 --> y_k
        if any(np.isnan(meas_diff[k, :])):
            continue
        _cost += meas_diff[k, :].T @ np.linalg.inv(meas_model.meas_noise(k)) @ meas_diff[k, :]
    _cost += meas_diff[-1, :].T @ np.linalg.inv(meas_model.meas_noise(K)) @ meas_diff[-1, :]

    return _cost


def dir_der_analytical_smoothing_cost(
    x_0, p, measurements, m_1_0, P_1_0, motion_model: MotionModel, meas_model: MeasModel, K=None
):
    """Directional derivative of the cost function f in `analytical_smoothing_cost`

    Here, the full trajectory x_1:K is interpreted as one vector (x_1^T, ..., x_K)^T with K d_x elements.

    Args:
        x_0: current iterate
            represented as a np.array(K, D_x).
        p: search direction, here: x_1 - x_0, i.e. new smoothing estimated means minus current iterate.
            represented as a np.array(K, D_x).
        measurements: measurements for a time sequence 1, ..., K
            represented as a list of length K of np.array(D_y,)
    """
    if K is None:
        K = len(measurements)

    prior_diff = x_0[0, :] - m_1_0
    motion_diff = x_0[1:, :] - motion_model.map_set(x_0[:-1, :], None)
    meas_diff = measurements - meas_model.map_set(x_0, None)

    der = p[0, :] @ np.linalg.inv(P_1_0) @ prior_diff
    H_1 = meas_model.jacobian(x_0[0, :], 1)
    R_1_inv = np.linalg.inv(meas_model.meas_noise(1))
    der -= p[0, :] @ H_1.T @ R_1_inv @ meas_diff[0, :]

    for k_ind in range(1, K):
        k = k_ind + 1
        F_k_min_1 = motion_model.jacobian(x_0[k_ind - 1, :], k - 1)
        factor_1 = p[k_ind, :].T - F_k_min_1 @ p[k_ind - 1, :].T
        der += factor_1 @ np.linalg.inv(motion_model.proc_noise(k - 1)) @ motion_diff[k_ind - 1, :]
        if any(np.isnan(meas_diff[k_ind, :])):
            continue
        H_k = meas_model.jacobian(x_0[k_ind, :], k)
        R_k_inv = np.linalg.inv(meas_model.meas_noise(k))
        der -= p[k_ind, :] @ H_k.T @ R_k_inv @ meas_diff[k_ind, :]

    return der


def grad_analytical_smoothing_cost(x_0, measurements, m_1_0, P_1_0, motion_model: MotionModel, meas_model: MeasModel):
    """Gradient of the cost function f in `analytical_smoothing_cost`
    Here, the full trajectory x_1:K is interpreted as one vector (x_1^T, ..., x_K)^T with K d_x elements.

    Args:
        x_0: current iterate
            represented as a np.array(K, D_x).
        measurements: measurements for a time sequence 1, ..., K
            represented as a list of length K of np.array(D_y,)
    """
    K = len(measurements)
    D_x = m_1_0.shape[0]
    grad_pred = np.zeros((K * D_x,))
    grad_update = np.zeros(grad_pred.shape)

    prior_diff = x_0[0, :] - m_1_0
    motion_diff = x_0[1:, :] - motion_model.map_set(x_0[:-1, :], None)

    grad_pred[0:D_x] = np.linalg.inv(P_1_0) @ prior_diff
    F_1 = motion_model.jacobian(x_0[0, :], 1)
    grad_update[0:D_x] = F_1.T @ np.linalg.inv(motion_model.proc_noise(1)) @ motion_diff[0, :]

    for k_ind in range(1, K - 1):
        k = k_ind + 1
        grad_pred[k_ind * D_x : (k_ind + 1) * D_x] = (
            np.linalg.inv(motion_model.proc_noise(k - 1)) @ motion_diff[k_ind - 1, :]
        )
        F_k = motion_model.jacobian(x_0[k_ind, :], k)
        grad_update[k_ind * D_x : (k_ind + 1) * D_x] = (
            F_k.T @ np.linalg.inv(motion_model.proc_noise(k)) @ motion_diff[k_ind, :]
        )

    grad_update[0:D_x] = np.linalg.inv(motion_model.proc_noise(K - 1)) @ motion_diff[-1, :]

    meas_diff = measurements - meas_model.map_set(x_0, None)
    grad_meas = np.zeros(grad_pred.shape)
    for k_ind in range(0, K):
        k = k_ind + 1
        if any(np.isnan(meas_diff[k_ind, :])):
            continue
        H_k = meas_model.jacobian(x_0[k_ind, :], k)
        R_k_inv = np.linalg.inv(meas_model.meas_noise(k))
        grad_meas[k_ind * D_x : (k_ind + 1) * D_x] = H_k.T @ R_k_inv @ meas_diff[k_ind, :]

    return grad_pred - grad_meas  # - grad_update


def analytical_smoothing_cost_time_dep(
    traj, measurements, m_1_0, P_1_0, motion_model: MotionModel, meas_model: MeasModel
):
    """Cost function for an optimisation problem used in the family of extended smoothers
    General formulation which does not assume that the motion and meas models is the same for all time steps.

    GN optimisation of this cost function will result in a linearised function
    corresponding to the Iterated Extended Kalman Smoother (IEKS).

    Args:
        traj: states for a time sequence 1, ..., K
            represented as a np.array(K, D_x).
            (The actual variable in the cost function)
        measurements: measurements for a time sequence 1, ..., K
            represented as a list of length K of np.array(D_y,)
    """
    prior_diff = traj[0, :] - m_1_0
    _cost = prior_diff.T @ np.linalg.inv(P_1_0) @ prior_diff

    K, D_x = traj.shape
    for k in range(1, K + 1):
        k_ind = k - 1
        if k < K:
            motion_diff_k = traj[k_ind + 1, :] - motion_model.mapping(traj[k_ind, :], k)
            _cost += motion_diff_k.T @ np.linalg.inv(motion_model.proc_noise(k)) @ motion_diff_k
        meas_k = measurements[k_ind]
        if any(np.isnan(meas_k)):
            continue
        meas_diff_k = meas_k - meas_model.mapping(traj[k_ind, :], k)
        _cost += meas_diff_k.T @ np.linalg.inv(meas_model.meas_noise(k)) @ meas_diff_k

    return _cost


def analytical_smoothing_cost_lm_ext(
    traj, measurements, prev_means, m_1_0, P_1_0, motion_model: MotionModel, meas_model: MeasModel, lambda_
):
    """Cost function for an optimisation problem used in the family of extended smoothers
    with LM regularisation

    GN optimisation of this cost function will result in a linearised function
    corresponding to the Levenberg-Marquardt Iterated Extended Kalman Smoother (IEKS)

    Args:
        traj: states for a time sequence 1, ..., K
            represented as a np.array(K, D_x).
            (The actual variable in the cost function)
        measurements: measurements for a time sequence 1, ..., K
            represented as a list of length K of np.array(D_y,)
    """
    prior_diff = traj[0, :] - m_1_0
    _cost = prior_diff.T @ np.linalg.inv(P_1_0) @ prior_diff

    motion_diff = traj[1:, :] - motion_model.map_set(traj[:-1, :], None)
    meas_diff = measurements - meas_model.map_set(traj, None)
    for k in range(0, traj.shape[0] - 1):
        _cost += motion_diff[k, :].T @ np.linalg.inv(motion_model.proc_noise(k)) @ motion_diff[k, :]
        # measurements are zero indexed, i.e. k-1 --> y_k
        if any(np.isnan(meas_diff[k, :])):
            continue
        _cost += meas_diff[k, :].T @ np.linalg.inv(meas_model.meas_noise(k)) @ meas_diff[k, :]
    _cost += meas_diff[-1, :].T @ np.linalg.inv(meas_model.meas_noise(measurements.shape[0])) @ meas_diff[-1, :]

    lm_dist = _lm_ext(traj, prev_means, lambda_)
    _cost += lm_dist

    return _cost


def _lm_ext(x, prev_x, lambda_):
    return lambda_ * ((x - prev_x) ** 2).sum()


def _ss_cost(means, measurements, m_1_0, P_1_0, Q, R, f_fun, h_fun):
    """Direct port of Simo Särkkä's matlab cost fn

    Only kept here for debugging purposes
    """
    J = (means[0, :] - m_1_0) @ np.linalg.inv(P_1_0) @ (means[0, :] - m_1_0)
    for k in range(0, means.shape[0]):
        x_k = means[k, :]
        z_k = measurements[k, :]
        if k > 0:
            x_k_min_1 = means[k - 1, :]
            J += (x_k - f_fun(x_k_min_1)).T @ np.linalg.inv(Q) @ (x_k - f_fun(x_k_min_1))
        J += (z_k - h_fun(x_k)).T @ np.linalg.inv(R) @ (z_k - h_fun(x_k))
    return J

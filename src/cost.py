"""Cost functions

TODO: The models are assumed to be indep of time step k.
Should make the API more flexible, this will prevent some vectorisation but this is not a hot path anyway.
"""
import logging
from functools import partial
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
            represented as a np.array(K, D_y)
    """
    K = len(measurements)
    prior_diff = traj[0, :] - m_1_0
    _cost = prior_diff.T @ np.linalg.inv(P_1_0) @ prior_diff

    proc_diff = traj[1:, :] - motion_model.map_set(traj[:-1, :], None)
    meas_diff = measurements - meas_model.map_set(traj, None)
    for k in range(0, K - 1):
        _cost += proc_diff[k, :].T @ np.linalg.inv(motion_model.proc_noise(k)) @ proc_diff[k, :]
        # measurements are zero indexed, i.e. k-1 --> y_k
        if any(np.isnan(meas_diff[k, :])):
            continue
        _cost += meas_diff[k, :].T @ np.linalg.inv(meas_model.meas_noise(k)) @ meas_diff[k, :]
    _cost += meas_diff[-1, :].T @ np.linalg.inv(meas_model.meas_noise(K)) @ meas_diff[-1, :]

    return _cost


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
            represented as a np.array(K, D_y)
    """
    prior_diff = traj[0, :] - m_1_0
    _cost = prior_diff.T @ np.linalg.inv(P_1_0) @ prior_diff

    K, D_x = traj.shape
    _, D_y = measurements.shape
    proc_diff = np.empty((K - 1, D_x))
    meas_diff = np.empty((K, D_y))
    # TODO: collapse into singel loop.
    for k in range(1, K + 1):
        k_ind = k - 1
        if k < K:
            proc_diff[k_ind, :] = traj[k, :] - motion_model.mapping(traj[k - 1, :], k - 1)
        meas_diff[k_ind, :] = measurements[k_ind, :] - meas_model.mapping(traj[k_ind, :], k)
    for k in range(0, K - 1):
        _cost += proc_diff[k, :].T @ np.linalg.inv(motion_model.proc_noise(k)) @ proc_diff[k, :]
        # measurements are zero indexed, i.e. k-1 --> y_k
        if any(np.isnan(meas_diff[k, :])):
            continue
        _cost += meas_diff[k, :].T @ np.linalg.inv(meas_model.meas_noise(k)) @ meas_diff[k, :]
    _cost += meas_diff[-1, :].T @ np.linalg.inv(meas_model.meas_noise(measurements.shape[0])) @ meas_diff[-1, :]

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
            represented as a np.array(K, D_y)
    """
    prior_diff = traj[0, :] - m_1_0
    _cost = prior_diff.T @ np.linalg.inv(P_1_0) @ prior_diff

    proc_diff = traj[1:, :] - motion_model.map_set(traj[:-1, :], None)
    meas_diff = measurements - meas_model.map_set(traj, None)
    for k in range(0, traj.shape[0] - 1):
        _cost += proc_diff[k, :].T @ np.linalg.inv(motion_model.proc_noise(k)) @ proc_diff[k, :]
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


def slr_smoothing_cost_pre_comp(traj, measurements, m_1_0, P_1_0, proc_bar, meas_bar, proc_cov, meas_cov):
    """Cost function for an optimisation problem used in the family of slr smoothers

    GN optimisation of this cost function will result in a linearised function
    corresponding to the SLR Smoother (PrLS, PLS) et al.

    Args:
        traj: states for a time sequence 1, ..., K
            represented as a np.array(K, D_x).
            (The actual variable in the cost function)
        covs: estimated covariances (means) for a time sequence 1, ..., K
            represented as a np.array(K, D_x, D_x)
        meas: measurements for a time sequence 1, ..., K
            represented as a np.array(K, D_y)
    """
    prior_diff = traj[0, :] - m_1_0
    _cost = prior_diff.T @ np.linalg.inv(P_1_0) @ prior_diff

    proc_diff = traj[1:, :] - proc_bar[:-1, :]
    meas_diff = measurements - meas_bar
    # print(f"Fast: {proc_diff[0, :].T @ np.linalg.inv(proc_cov[0, :, :]) @ proc_diff[0, :]}")

    for k in range(0, traj.shape[0] - 1):
        _cost += proc_diff[k, :].T @ np.linalg.inv(proc_cov[k, :, :]) @ proc_diff[k, :]
        if any(np.isnan(meas_diff[k, :])):
            continue
        # measurements are zero indexed, i.e. k-1 --> y_k
        _cost += meas_diff[k, :].T @ np.linalg.inv(meas_cov[k, :, :]) @ meas_diff[k, :]
    _cost += meas_diff[-1, :].T @ np.linalg.inv(meas_cov[-1, :, :]) @ meas_diff[-1, :]

    return _cost


def slr_smoothing_cost(
    traj,
    covs,
    measurements,
    m_1_0,
    P_1_0,
    motion_model: MotionModel,
    meas_model: MeasModel,
    slr: Slr,
):
    """Cost function for an optimisation problem used in the family of slr smoothers

    GN optimisation of this cost function will result in a linearised function
    corresponding to the SLR Smoother (PrLS, PLS) et al.

    Args:
        traj: states for a time sequence 1, ..., K
            represented as a np.array(K, D_x).
            (The actual variable in the cost function)
        covs: estimated covariances (means) for a time sequence 1, ..., K
            represented as a np.array(K, D_x, D_x)
        meas: measurements for a time sequence 1, ..., K
            represented as a np.array(K, D_y)
    """
    prior_diff = traj[0, :] - m_1_0
    _cost = prior_diff.T @ np.linalg.inv(P_1_0) @ prior_diff

    motion_mapping = partial(motion_model.map_set, time_step=None)
    for k in range(0, traj.shape[0] - 1):
        mean_k = traj[k, :]
        cov_k = covs[k, :, :]
        proc_bar, psi, phi = slr.slr(motion_mapping, mean_k, cov_k)
        _, _, Omega_k = slr.linear_params_from_slr(mean_k, cov_k, proc_bar, psi, phi)

        proc_diff_k = traj[k + 1, :] - proc_bar
        _cost += proc_diff_k.T @ np.linalg.inv(motion_model.proc_noise(k) + Omega_k) @ proc_diff_k

    meas_mapping = partial(meas_model.map_set, time_step=None)
    for k in range(0, traj.shape[0]):
        if any(np.isnan(measurements[k, :])):
            continue
        mean_k = traj[k, :]
        cov_k = covs[k, :]
        meas_bar, psi, phi = slr.slr(meas_mapping, mean_k, cov_k)
        _, _, Lambda_k = slr.linear_params_from_slr(mean_k, cov_k, meas_bar, psi, phi)

        # measurements are zero indexed, i.e. meas[k-1] --> y_k
        meas_diff_k = measurements[k, :] - meas_bar
        _cost += meas_diff_k.T @ np.linalg.inv(meas_model.meas_noise(k) + Lambda_k) @ meas_diff_k

    return _cost


def noop_cost(means, covs):
    LOGGER.warning("Using the dummy loss")
    return None


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

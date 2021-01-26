"""Cost functions

TODO: The models are assumed to be indep of time step k.
Should make the API more flexible, this will prevent some vectorisation but this is not a hot path anyway.
"""
import logging
import numpy as np
from src.models.base import MotionModel, MeasModel
from src.slr.base import Slr

LOGGER = logging.getLogger(__name__)


def analytical_smoothing_cost(traj, meas, m_1_0, P_1_0, motion_model: MotionModel, meas_model: MeasModel):
    """Cost function for an optimisation problem used in the family of extended smoothers

    GN optimisation of this cost function will result in a linearised function
    corresponding to the Extended Kalman Smoother (EKS) et al.

    Args:
        traj: states for a time sequence 1, ..., K
            represented as a np.array(K, D_x).
            (The actual variable in the cost function)
        meas: measurements for a time sequence 1, ..., K
            represented as a np.array(K, D_y)
    """
    prior_diff = traj[0, :] - m_1_0
    _cost = prior_diff.T @ np.linalg.inv(P_1_0) @ prior_diff

    proc_diff = traj[1:, :] - motion_model.map_set(traj[:-1, :])
    meas_diff = meas - meas_model.map_set(traj)
    for k in range(0, traj.shape[0] - 1):
        _cost += proc_diff[k, :].T @ np.linalg.inv(motion_model.proc_noise(k)) @ proc_diff[k, :]
        # measurements are zero indexed, i.e. k-1 --> y_k
        _cost += meas_diff[k, :].T @ np.linalg.inv(meas_model.meas_noise(k)) @ meas_diff[k, :]
    _cost += meas_diff[-1, :].T @ np.linalg.inv(meas_model.meas_noise(k)) @ meas_diff[-1, :]

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
            represented as a np.array(K, D_x)
        meas: measurements for a time sequence 1, ..., K
            represented as a np.array(K, D_y)
    """
    prior_diff = traj[0, :] - m_1_0
    _cost = prior_diff.T @ np.linalg.inv(P_1_0) @ prior_diff

    for k in range(0, traj.shape[0] - 1):
        mean_k = traj[k, :]
        cov_k = covs[k, :]
        proc_bar, _, _ = slr.slr(motion_model.map_set, mean_k, cov_k)
        _, _, Omega_k = slr.linear_params(motion_model.map_set, mean_k, cov_k)

        proc_diff_k = traj[k + 1, :] - proc_bar
        _cost += proc_diff_k.T @ np.linalg.inv(motion_model.proc_noise(k) + Omega_k) @ proc_diff_k

    for k in range(0, traj.shape[0]):
        mean_k = traj[k, :]
        cov_k = covs[k, :]
        meas_bar, _, _ = slr.slr(meas_model.map_set, mean_k, cov_k)
        _, _, Lambda_k = slr.linear_params(meas_model.map_set, mean_k, cov_k)

        # measurements are zero indexed, i.e. meas[k-1] --> y_k
        meas_diff_k = measurements[k, :] - meas_bar
        _cost += meas_diff_k.T @ np.linalg.inv(meas_model.meas_noise(k)) @ meas_diff_k

    return _cost


def noop_cost(traj):
    LOGGER.info("Using the dummy loss")
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

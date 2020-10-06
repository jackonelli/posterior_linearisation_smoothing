"""Kalman filter (KF)"""
import logging
import numpy as np
from src.linearizer import Linearizer
from src.slr.distributions import Prior, Conditional
from src.slr.slr import Slr

LOGGER = logging.getLogger(__name__)


def kalman_filter(
    measurements,
    x_0_0,
    P_0_0,
    motion_lin: Linearizer,
    meas_lin: Linearizer,
):
    """Kalman filter with general linearization
    Filters a measurement sequence using a linear Kalman filter.
    Linearization is done with a general type
    implementing the abstract class Linearizer

    Args:
        measurements (K, D_y): Measurement sequence for times 1,..., K
        x_0_0 (D_x,): Prior mean for time 0
        P_0_0 (D_x, D_x): Prior covariance for time 0
        motion_lin: Must inherit from Linearizer
        meas_lin: Must inherit from Linearizer

    Returns:
        filter_means (K, D_x): Filtered estimates for times 1,..., K
        filter_covs (K, D_x, D_x): Filter error covariance
        pred_means (K, D_x): Predicted estimates for times 1,..., K
        pred_covs (K, D_x, D_x): Filter error covariance
        linearizations List(np.array, np.array, np.array):
            List of tuples (A, b, Q), param's for linear approx of the motion model.
    """

    K = measurements.shape[0]

    filter_means, filter_covs = _init_estimates(x_0_0, P_0_0, K)
    pred_means, pred_covs = _init_estimates(x_0_0, P_0_0, K)
    linearizations = [None] * K

    x_kminus1_kminus1 = x_0_0
    P_kminus1_kminus1 = P_0_0
    for k in np.arange(1, K + 1):
        LOGGER.debug("Time step: %s", k)
        # measurment vec is zero-indexed
        # this really gives y_k
        y_k = measurements[k - 1]
        motion_lin_kminus1 = motion_lin.linear_params(x_kminus1_kminus1, P_kminus1_kminus1)
        x_k_kminus1, P_k_kminus1 = _predict(x_kminus1_kminus1, P_kminus1_kminus1, motion_lin_kminus1)

        meas_lin_k = meas_lin.linear_params(x_k_kminus1, P_k_kminus1)
        x_k_k, P_k_k = _update(y_k, x_k_kminus1, P_k_kminus1, meas_lin_k)

        linearizations[k - 1] = motion_lin_kminus1
        pred_means[k, :] = x_k_kminus1
        pred_covs[k, :, :] = P_k_kminus1
        filter_means[k, :] = x_k_k
        filter_covs[k, :, :] = P_k_k
        # Shift to next time step
        x_kminus1_kminus1 = x_k_k
        P_kminus1_kminus1 = P_k_k

    return filter_means, filter_covs, pred_means, pred_covs, linearizations


def kalman_filter_known_post(
    measurements,
    x_0_0,
    P_0_0,
    prev_smooth_means,
    prev_smooth_covs,
    motion_lin,
    meas_lin,
):
    """Kalman filter with general linearization
    Filters a measurement sequence using a linear Kalman filter.
    The linearization is done w.r.t. to a known posterior estimate.
    Linearization is done with a general type
    implementing the abstract class Linearizer

    Args:
        measurements (K, D_y): Measurement sequence for times 1,..., K
        x_0_0 (D_x,): Prior mean for time 0
        P_0_0 (D_x, D_x): Prior covariance for time 0
        motion_lin: Must inherit from Linearizer
        meas_lin: Must inherit from Linearizer

    Returns:
        filter_means (K, D_x): Filtered estimates for times 1,..., K
        filter_covs (K, D_x, D_x): Filter error covariance
        pred_means (K, D_x): Predicted estimates for times 1,..., K
        pred_covs (K, D_x, D_x): Filter error covariance
        linearizations List(np.array, np.array, np.array):
            List of tuples (A, b, Q), param's for linear approx of the motion model.
    """

    K = measurements.shape[0]

    filter_means, filter_covs = _init_estimates(x_0_0, P_0_0, K)
    pred_means, pred_covs = _init_estimates(x_0_0, P_0_0, K)
    linearizations = [None] * K

    x_kminus1_kminus1 = x_0_0
    P_kminus1_kminus1 = P_0_0
    for k in np.arange(1, K + 1):
        # measurment vec is zero-indexed
        # this really gives y_k
        y_k = measurements[k - 1]
        x_kminus1_K = prev_smooth_means[k - 1, :]
        P_kminus1_K = prev_smooth_covs[k - 1, :, :]
        x_k_K = prev_smooth_means[k, :]
        P_k_K = prev_smooth_covs[k, :, :]
        motion_lin_kminus1 = motion_lin.linear_params(x_kminus1_K, P_kminus1_K)
        x_k_kminus1, P_k_kminus1 = _predict(x_kminus1_kminus1, P_kminus1_kminus1, motion_lin_kminus1)

        meas_lin_k = meas_lin.linear_params(x_k_K, P_k_K)
        x_k_k, P_k_k = _update(y_k, x_k_kminus1, P_k_kminus1, meas_lin_k)

        linearizations[k - 1] = motion_lin_kminus1
        pred_means[k, :] = x_k_kminus1
        pred_covs[k, :, :] = P_k_kminus1
        filter_means[k, :] = x_k_k
        filter_covs[k, :, :] = P_k_k
        # Shift to next time step
        x_kminus1_kminus1 = x_k_k
        P_kminus1_kminus1 = P_k_k
    return filter_means, filter_covs, pred_means, pred_covs, linearizations


def _init_estimates(x_0_0, P_0_0, K):
    D_x = x_0_0.shape[0]
    est_means = np.empty((K + 1, D_x))
    est_covs = np.empty((K + 1, D_x, D_x))
    est_means[0, :] = x_0_0
    est_covs[0, :, :] = P_0_0
    return est_means, est_covs


def analytical_kf(measurements, x_0_0, P_0_0, motion_lin, meas_lin):
    """SLR Kalman filter with SLR linearization
    Filters a measurement sequence using a linear Kalman filter.
    Linearization is done with SLR
    Args:
        measurements np.array(K, D_y): Measurement sequence for times 1,..., K
        x_0_0 np.array(D_x,): Prior mean for time 0
        P_0_0 np.array(D_x, D_x): Prior covariance
        motion_lin: Param's for lin. (affine) transf: (A, b, Q)
        meas_lin: Param's for lin. (affine) transf: (H, c, R)

    Returns:
        filter_means np.array(K, D_x): Filtered estimates for times 1,..., K
        filter_covs np.array(K, D_x, D_x): Filter error covariance
        pred_means np.array(): Predicted estimates for times 1,..., K
        pred_covs np.array(): Filter error covariance
        linearizations List(np.array, np.array, np.array):
            List of tuples (A, b, Q), param's for linear approx
    """

    K = measurements.shape[0]
    D_x = x_0_0.shape[0]

    filter_means = np.empty((K, D_x))
    filter_covs = np.empty((K, D_x, D_x))
    pred_means = np.empty((K, D_x))
    pred_covs = np.empty((K, D_x, D_x))
    x_kminus1_kminus1 = x_0_0
    P_kminus1_kminus1 = P_0_0
    for k in np.arange(1, K + 1):
        y_k = measurements[k - 1]

        x_k_kminus1, P_k_kminus1 = _predict(x_kminus1_kminus1, P_kminus1_kminus1, motion_lin)

        x_k_k, P_k_k = _update(y_k, x_k_kminus1, P_k_kminus1, meas_lin)

        pred_means[k, :] = x_k_kminus1
        pred_covs[k, :, :] = P_k_kminus1
        filter_means[k, :] = x_k_k
        filter_covs[k, :, :] = P_k_k
        x_0_0 = x_k_k
        P_0_0 = P_k_k
    return filter_means, filter_covs, pred_means, pred_covs


def _predict(x_kminus1_kminus1, P_kminus1_kminus1, linearization):
    """KF prediction step

    Args:
        x_kminus1_kminus1: x_{k-1 | k-1}
        P_kminus1_kminus1: P_{k-1 | k-1}
        linearization (tuple): (A, b, Q) param's for linear (affine) approx

    Returns:
        x_k_kminus1: x_{k | k-1}
        P_k_kminus1: P_{k | k-1}
    """
    A, b, Q = linearization
    x_k_kminus1 = A @ x_kminus1_kminus1 + b
    P_k_kminus1 = A @ P_kminus1_kminus1 @ A.T + Q
    P_k_kminus1 = (P_k_kminus1 + P_k_kminus1.T) / 2
    return x_k_kminus1, P_k_kminus1


def _update(y_k, x_k_kminus1, P_k_kminus1, linearization):
    """KF update step
    Args:
        y_k
        x_k_kminus1: x_{k | k-1}
        P_k_kminus1: P_{k | k-1}
        linearization (tuple): (A, b, Q) param's for linear (affine) approx

    Returns:
        x_k_k: x_{k | k}
        P_k_k: P_{k | k}
    """
    H, c, R = linearization
    y_mean = H @ x_k_kminus1 + c
    S = H @ P_k_kminus1 @ H.T + R
    K = P_k_kminus1 @ H.T @ np.linalg.inv(S)

    x_k_k = x_k_kminus1 + (K @ (y_k - y_mean)).reshape(x_k_kminus1.shape)
    P_k_k = P_k_kminus1 - K @ S @ K.T

    return x_k_k, P_k_k

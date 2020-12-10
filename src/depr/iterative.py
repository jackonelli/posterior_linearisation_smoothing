"""Iterative wrapper for the posterior linearization smoother"""
import logging
import numpy as np
from src.filter.base import Filter
from src.smoother.base import Smoother

LOGGER = logging.getLogger(__name__)


def iterative_post_lin_smooth(
    measurements,
    x_0_0,
    P_0_0,
    filter_: Filter,
    smoother: Smoother,
    num_iterations: int,
):
    """Iterative posterior linearization smoothing
    First iteration performs Kalman filtering with SLR and RTS smoothing
    Subsequent iterations use smooth estimates from previous iteration
    in the linearization.

    TODO: Remove linearization as an output. Only used for debug.

    Args:
        measurements (K, D_y)
        x_0_0 (D_x,): Prior mean for time 0
        P_0_0 (D_x, D_x): Prior covariance for time 0
        prior: p(x) used in SLR.
            Note that this is given as a class prototype,
            it is instantiated multiple times in the function
        motion_model: p(x_k+1 | x_k) used in SLR
        meas_model: p(y_k | x_k) used in SLR
        num_samples
        num_iterations
    """
    LOGGER.info("Smoothing iter: %d", 1)
    (smooth_means, smooth_covs, filter_means, filter_covs) = _first_iter(measurements, x_0_0, P_0_0, filter_, smoother)
    for iter_ in np.arange(2, num_iterations + 1):
        LOGGER.info("Smoothing iter: %d", iter_)
        (smooth_means, smooth_covs, filter_means, filter_covs) = _iteration(
            measurements, x_0_0, P_0_0, smooth_means, smooth_covs, filter_, smoother
        )
    return smooth_means, smooth_covs, filter_means, filter_covs


def _first_iter(measurements, x_0_0, P_0_0, filter_, smoother):
    """First iteration
    Special case since no smooth estimates exist from prev iteration
    Performs KF with gen. linearization, then RTS smoothing.
    """
    filter_means, filter_covs, pred_means, pred_covs, linearizations = filter_.filter_seq(
        measurements,
        x_0_0,
        P_0_0,
    )
    smooth_means, smooth_covs = smoother.smooth_seq(filter_means, filter_covs, pred_means, pred_covs)
    return smooth_means, smooth_covs, filter_means, filter_covs, linearizations


def _iteration(measurements, x_0_0, P_0_0, prev_smooth_means, prev_smooth_covs, filter_, smoother):
    """General non-first iteration
    Performs KF but uses smooth estimates from prev iteration as priors in
    the linearization.
    """
    (filter_means, filter_covs, pred_means, pred_covs, linearizations) = kalman_filter_known_post(
        measurements, x_0_0, P_0_0, prev_smooth_means, prev_smooth_covs, motion_lin, meas_lin
    )

    smooth_means, smooth_covs = rts_smoothing(filter_means, filter_covs, pred_means, pred_covs, linearizations)
    return smooth_means, smooth_covs, filter_means, filter_covs, linearizations

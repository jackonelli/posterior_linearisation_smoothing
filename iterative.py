"""Iterative wrapper for the posterior linearization smoother"""
import numpy as np
import matplotlib.pyplot as plt
from post_lin_smooth.slr.distributions import Prior, Conditional
from post_lin_smooth.filtering import kalman_filter, kalman_filter_known_post
from post_lin_smooth.smoothing import rts_smoothing
from post_lin_smooth.linearizer import Linearizer


def iterative_post_lin_smooth(measurements,
                              x_0_0,
                              P_0_0,
                              motion_lin: Linearizer,
                              meas_lin: Linearizer,
                              num_iterations: int,
                              normalize=False):
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
    print("Iter: ", 1)
    (smooth_means,
     smooth_covs,
     filter_means,
     filter_covs,
     linearizations) = _first_iter(measurements,
                                   x_0_0,
                                   P_0_0,
                                   motion_lin,
                                   meas_lin,
                                   normalize=False)
    for iter_ in np.arange(2, num_iterations + 1):
        print("Iter: ", iter_)
        (smooth_means,
         smooth_covs,
         filter_means,
         filter_covs,
         linearizations) = _iteration(measurements,
                                      x_0_0,
                                      P_0_0,
                                      smooth_means,
                                      smooth_covs,
                                      motion_lin,
                                      meas_lin)
    return smooth_means, smooth_covs, filter_means, filter_covs, linearizations


def _first_iter(measurements, x_0_0, P_0_0, motion_lin, meas_lin):
    """First iteration
    Special case since no smooth estimates exist from prev iteration
    Performs KF with gen. linearization, then RTS smoothing.
    """
    filter_means, filter_covs, pred_means, pred_covs, linearizations = kalman_filter(
        measurements, x_0_0, P_0_0, motion_lin, meas_lin, normalize=normalize)
    smooth_means, smooth_covs = rts_smoothing(filter_means, filter_covs,
                                              pred_means, pred_covs,
                                              linearizations)
    return smooth_means, smooth_covs, filter_means, filter_covs, linearizations


def _iteration(measurements,
               x_0_0,
               P_0_0,
               prev_smooth_means,
               prev_smooth_covs,
               motion_lin,
               meas_lin):
    """General non-first iteration
    Performs KF but uses smooth estimates from prev iteration as priors in
    the linearization.
    Standard RTS
    """
    (filter_means,
     filter_covs,
     pred_means,
     pred_covs,
     linearizations) = kalman_filter_known_post(measurements,
                                                x_0_0,
                                                P_0_0,
                                                prev_smooth_means,
                                                prev_smooth_covs,
                                                motion_lin,
                                                meas_lin)

    smooth_means, smooth_covs = rts_smoothing(filter_means, filter_covs,
                                              pred_means, pred_covs,
                                              linearizations)
    return smooth_means, smooth_covs, filter_means, filter_covs, linearizations

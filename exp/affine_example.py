"""Example: Iterative post. lin. smoothing with affine models"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from src import visualization as vis
from src.filter.kalman import KalmanFilter
from src.smoother.rts import RtsSmoother
from src.utils import setup_logger
from src.models.affine import AffineModel


def main():
    log = logging.getLogger(__name__)
    experiment_name = "affine_problem"
    setup_logger(f"logs/{experiment_name}.log", logging.INFO)
    log.info(f"Running experiment: {experiment_name}")
    K = 20

    prior_mean = np.array([1, 1, 3, 2])
    prior_cov = 1 * np.eye(4)
    T = 1
    A = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]])
    b = 0 * np.ones((4,))
    Q = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1.5, 0],
            [0, 0, 0, 1.5],
        ]
    )
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    c = np.zeros((H @ prior_mean).shape)
    R = 2 * np.eye(2)

    motion_lin = AffineModel(A, b, Q)
    meas_lin = AffineModel(H, c, R)
    true_x = gen_linear_state_seq(prior_mean, prior_cov, A, Q, K)
    y = gen_linear_meas_seq(true_x, H, R)

    analytical_smooth = RtsSmoother(motion_lin, meas_lin)
    xf, Pf, xs, Ps = analytical_smooth.filter_and_smooth(y, prior_mean, prior_cov)
    vis.plot_nees_and_2d_est(true_x, y, xf, Pf, xs, Ps, sigma_level=3, skip_cov=2)


def gen_linear_state_seq(x_0, P_0, A, Q, K):
    """Generates an K-long sequence of states using a
    Gaussian prior and a linear Gaussian process model

    Args:
       x_0         [n x 1] Prior mean
       P_0         [n x n] Prior covariance
       A           [n x n] State transition matrix
       Q           [n x n] Process noise covariance
       K           [1 x 1] Number of states to generate

    Returns:
       X           [n x K+1] State vector sequence
    """
    X = np.zeros((K, x_0.shape[0]))

    X[0, :] = mvn.rvs(mean=A @ x_0, cov=P_0, size=1)

    q = mvn.rvs(mean=np.zeros(x_0.shape), cov=Q, size=K)

    for k in np.arange(1, K):
        X[k, :] = A @ X[k - 1, :] + q[k - 1, :]
    return X


def gen_linear_meas_seq(X, H, R):
    """generates a sequence of observations of the state
    sequence X using a linear measurement model.
    Measurement noise is assumed to be zero mean and Gaussian.

    Args:
        X [K x n] State vector sequence. The k:th state vector is X(k, :)
        H [m x n] Measurement matrix
        R [m x m] Measurement noise covariance

    Returns:
        Y [K, m] Measurement sequence
    """

    r = mvn.rvs(mean=np.zeros((R.shape[0],)), cov=R, size=X.shape[0])

    return (H @ X.T).T + r


if __name__ == "__main__":
    main()

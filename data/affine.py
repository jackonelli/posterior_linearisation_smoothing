import numpy as np
from src.models.affine import AffineModel
from scipy.stats import multivariate_normal as mvn


def sim_affine_state_seq(x_0: np.ndarray, P_0: np.ndarray, aff_model: AffineModel, K: int):
    """Generates an K-long sequence of states using a
    Gaussian prior and a linear Gaussian process model

    Args:
       x_0: Prior mean, (n,)
       P_0: Prior covariance, (n, n)
       A: State transition matrix, (n, n)
       Q: Process noise covariance, (n, n)
       K: Number of states to generate

    Returns:
       X           [n x K+1] State vector sequence
    """
    X = np.zeros((K, x_0.shape[0]))

    X[0, :] = mvn.rvs(mean=aff_model.mapping(x_0), cov=P_0, size=1)

    q = mvn.rvs(mean=np.zeros(x_0.shape), cov=aff_model.noise, size=K)

    for k in np.arange(1, K):
        X[k, :] = aff_model.mapping(X[k - 1, :]) + q[k - 1, :]
    return X


def sim_affine_meas_seq(X, aff_model: AffineModel):
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

    r = mvn.rvs(mean=np.zeros((aff_model.offset.shape)), cov=aff_model.noise, size=X.shape[0])

    return aff_model.map_set(X) + r

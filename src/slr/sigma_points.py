"""Statistical linear regression (SLR) with sigma points"""
import numpy as np
from src.sigma_points import SigmaPointMethod
from src.slr.base import Slr


class SigmaPointSlr(Slr):
    def __init__(self, sigma_point_method: SigmaPointMethod):
        self.sigma_point_method = sigma_point_method

    def slr(self, fn, mean, cov):
        """Sigma point SLR
        Calculate z_bar, psi, phi from a (non-linear) function and estimated distribution
        TODO:
            - Test that the weighted cov's work as intended.
        """

        sigma_points, weights = self.sigma_points(mean, cov)
        transf_sigma_points = fn(sigma_points)
        z_bar = weighted_avg(transf_sigma_points, weights)
        psi = weighted_cov(sigma_points, mean, transf_sigma_points, z_bar, weights)
        phi = weighted_cov(transf_sigma_points, z_bar, transf_sigma_points, z_bar, weights)

        return z_bar, psi, phi

    def sigma_points(self, mean, cov):
        return self.sigma_point_method.sigma_points(mean, cov)


def weighted_avg(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted average

    Args:
        vectors: List of N vectors of dim. d represented as matrix R^(Nxd)
        weights: N weights represented as vector R^(N,)

    Returns:
        weighted average: average vector, represented as vector R^(d,)
    """

    return weights @ vectors


def weighted_cov(
    x_vecs: np.ndarray, x_bar: np.ndarray, y_vecs: np.ndarray, y_bar: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Weighted average
    The *_bar quantities can be calculated from *_vecs but they are often already calculated, so for efficiency we reuse them.

    Args:
        x_vecs: List of N vectors of dim. D_x represented as matrix R^(NxD_x)
        x_bar:  Average of x_vecs, represented as a vector R^(D_x,)
        y_vecs: List of N vectors of dim. D_y represented as matriy R^(NxD_y)
        y_bar:  Average of y_vecs, represented as a vector R^(D_y,)
        weights: N weights represented as np array R^(N,)

    Returns:
        weighted average: average vector, represented as np array R^(d,)
    """

    x_diff = x_vecs - x_bar
    y_diff = y_vecs - y_bar
    return (weights * x_diff.T) @ y_diff

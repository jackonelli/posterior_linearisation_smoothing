"""Statistical linear regression (SLR) with sigma points"""
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import sqrtm


class SigmaPointMethod(ABC):
    @abstractmethod
    def sigma_points(mean, cov):
        pass


class SphericalCubature(SigmaPointMethod):
    def sigma_points(self, mean, cov):
        D_x = mean.shape[0]
        sqrt_cov = sqrtm(cov)
        num_sigma_points = 2 * D_x

        sigma_points = np.empty((num_sigma_points, D_x))
        for dim in np.arange(0, D_x):
            sigma_points[2 * dim, :] = mean + np.sqrt(D_x) * sqrt_cov[:, dim]
            sigma_points[2 * dim + 1, :] = mean - np.sqrt(D_x) * sqrt_cov[:, dim]

        weights = np.ones((num_sigma_points,)) / num_sigma_points
        return sigma_points, weights


class SigmaPointSlrBase(ABC):
    @abstractmethod
    def _sigma_points(self, mean, cov):
        pass


class PosteriorSigmaPointSlr(SigmaPointSlrBase):
    def __init__(self, means, covs):
        self.means = means
        self.covs = covs


class SigmaPointSlr(SigmaPointSlrBase):
    def __init__(self, sigma_point_method=SphericalCubature()):
        self.sigma_point_method = sigma_point_method

    def linear_params(self, fn, mean, cov):
        """SLR sigma points linearization
        Args:
            fn: state mapping. In principle fn: R^n -> R^m,
                but in practice the function should handle batches of vectors of lenght N.
            mean: estimated state, mean in R^n
            cov: estimate state covaraiance, R^(n x n)
        """

        z_bar, psi, phi = self._sigma_point_slr(fn, mean, cov)
        A = psi.T @ np.linalg.inv(cov)
        b = z_bar - A @ mean
        Sigma = phi - A @ cov @ A.T
        return A, b, Sigma

    def _sigma_point_slr(self, fn, mean, cov):
        """Sigma point SLR
        Calculate z_bar, psi, phi from a (non-linear) function and estimated distribution
        TODO:
            - Test that the weighted cov's work as intended.
            - If built in np cov insufficient, check monte_carlo.py for vectorization.
        """

        sigma_points, weights = self._sigma_points(mean, cov)
        transf_sigma_points = fn(sigma_points)
        z_bar = weighted_avg(transf_sigma_points, weights)
        print("Z\n{}\n".format(transf_sigma_points))
        psi = weighted_cov(sigma_points, mean, transf_sigma_points, z_bar, weights)
        print("psi\n{}\n".format(psi))
        phi = weighted_cov(transf_sigma_points, z_bar, transf_sigma_points, z_bar, weights)

        return z_bar, psi, phi

    def _sigma_points(self, mean, cov):
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
        x_vecs: List of N vectors of dim. d_x represented as matrix R^(Nxd_x)
        x_bar:  Average of x_vecs, represented as a vector R^(d_x,)
        y_vecs: List of N vectors of dim. d_y represented as matriy R^(Nxd_y)
        y_bar:  Average of y_vecs, represented as a vector R^(d_y,)
        weights: N weights represented as np array R^(N,)

    Returns:
        weighted average: average vector, represented as np array R^(d,)
    """

    x_diff = x_vecs - x_bar
    y_diff = y_vecs - y_bar
    return (weights * x_diff.T) @ y_diff

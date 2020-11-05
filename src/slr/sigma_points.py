"""Statistical linear regression (SLR) with sigma points"""
from abc import ABC, abstractmethod
import numpy as np


class SigmaPointSlr:
    def __init__(self, sigma_point_method=SphericalCubature()):
        self.sigma_point_method = sigma_point_method

    def linear_params(self, fn, mean, cov):
        """SLR sigma points linearization
        Args:
            fn: state mapping. In principle fn: R^n -> R^m,
                but in practice the function should handle batches of vectors of lenght n.
            mean: estimated state, mean in R^n
            cov: estimate state covaraiance, R^(n x n)
        """

        z_bar, psi, phi = self._sigma_point_slr(fn, mean, cov, sigma_point_method)
        A = psi.T @ np.linalg.inv(cov)
        b = z_bar - A @ mean
        Sigma = phi - A @ cov @ A.T
        return A, b, Sigma

    def _sigma_point_slr(self, fn, mean, cov, sigma_point_method):
        """Sigma point SLR
        Calculate z_bar, psi, phi from a (non-linear) function and estimated distribution
        TODO:
            - Test that the weighted cov's work as intended.
            - If built in np cov insufficient, check monte_carlo.py for vectorization.
        """

        sigma_points, weights = self.sigma_point_method._sigma_points(mean, cov)
        transf_sigma_points = fn(sigma_points)
        z_bar = transf_sigma_points.average(weights=weights)
        diff_sigma_points = sigma_points - mean
        diff_transf_sigma_points = transf_sigma_points - z_bar
        psi = np.cov(m=diff_sigma_points, y=diff_transf_sigma_points, aweights=weights)
        phi = np.cov(diff_transf_sigma_points, aweights=weights)

        return z_bar, psi, phi


class SigmaPointMethod(ABC):
    @abstractmethod
    def _sigma_points(mean, cov):
        pass


class SphericalCubature(SigmaPointMethod):
    def _sigma_points(self, mean, cov):
        D_x = mean.shape[1]
        sqrt_cov = np.sqrtm(cov)
        num_sigma_points = 2 * D_x

        sigma_points = np.empty((num_sigma_points, D_x))
        for dim in np.arange(0, D_x):
            sigma_points[2 * dim, :] = mean + np.sqrt(D_x) * sqrt_cov[:, dim]
            sigma_points[2 * dim + 1, :] = mean - np.sqrt(D_x) * sqrt_cov[:, dim]

        weights = np.ones((num_sigma_points, 1)) / num_sigma_points
        return sigma_points, weights

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
        cov = np.atleast_2d(cov)
        sqrt_cov = sqrtm(cov)
        num_sigma_points = 2 * D_x

        sigma_points = np.empty((num_sigma_points, D_x))
        for dim in np.arange(0, D_x):
            sigma_points[2 * dim, :] = mean + np.sqrt(D_x) * sqrt_cov[:, dim]
            sigma_points[2 * dim + 1, :] = mean - np.sqrt(D_x) * sqrt_cov[:, dim]

        weights = np.ones((num_sigma_points,)) / num_sigma_points
        return sigma_points, weights


class UnscentedTransform(SigmaPointMethod):
    """Unscented transform

    Based on sec. 5.5 of "Bayesian filtering and smoothing (Simo Särkkä)"

    TODO: Precompute weights for eff.?
    """

    def __init__(self, alpha, beta, kappa):
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa

    def lambda_(self, D_x):
        return self._alpha ** 2 * (D_x + self._kappa) - D_x

    def weights(self, D_x):
        lambda_ = self.lambda_(D_x)
        w_m_0 = np.array([lambda_ / (D_x + lambda_)])
        w_c_0 = np.array([lambda_ / (D_x + lambda_) + (1 - self._alpha ** 2 + self._beta)])
        tmp_w = 1 / (2 * (np.arange(1, 2 * D_x) + lambda_))
        return np.concatenate((w_m_0, tmp_w)), np.concatenate((w_c_0, tmp_w))

    def sigma_points(self, mean, cov):
        D_x = mean.shape[0]
        cov = np.atleast_2d(cov)
        sqrt_cov = sqrtm(cov)
        lambda_ = self.lambda_(D_x)
        num_sigma_points = 2 * D_x + 1

        sigma_points = np.empty((num_sigma_points, D_x))
        sigma_points[0, :] = mean
        for dim in np.arange(D_x):
            sigma_points[dim + 1, :] = mean + np.sqrt(D_x + lambda_) * sqrt_cov[:, dim]
            sigma_points[D_x + dim + 1, :] = mean - np.sqrt(D_x + lambda_) * sqrt_cov[:, dim]

        weights = np.ones((num_sigma_points,)) / num_sigma_points
        return sigma_points, weights

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

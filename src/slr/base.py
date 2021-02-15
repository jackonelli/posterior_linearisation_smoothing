"""Statistical linear regression (SLR) with sigma points"""
from abc import ABC, abstractmethod
import numpy as np


class Slr(ABC):
    def linear_params(self, fn, mean, cov):
        """SLR linearisation
        Args:
            fn: state mapping. In principle fn: R^n -> R^m,
                but in practice the function must handle sets of vectors of length N.
                I.e., fn: R^Nxn -> R^Nxm
            mean: mean, R^n
            cov: covaraiance, R^(n x n)
        """

        mean, cov = np.atleast_1d(mean), np.atleast_2d(cov)
        # print("mean", mean.max())
        # print("cov", cov.max())
        z_bar, psi, phi = self.slr(fn, mean, cov)
        A = psi.T @ np.linalg.inv(cov)
        b = z_bar - A @ mean
        Sigma = phi - A @ cov @ A.T
        return A, b, Sigma

    @abstractmethod
    def slr(self, fn, mean, cov):
        pass

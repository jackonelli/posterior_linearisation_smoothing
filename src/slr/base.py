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
        z_bar, psi, phi = self.slr(fn, mean, cov)
        return self.linear_params_from_slr(mean, cov, z_bar, psi, phi)

    @staticmethod
    def linear_params_from_slr(mean, cov, z_bar, psi, phi):
        """SLR linearisation helper

        Public fn because sometimes the desired output is the SLR qty's in addition to the lin. params.
        Without this API that output would require calling both `self.slr` and then `self.linear_params`,
        causing unecc. sigma point calculation.
        With it, `self.linear_params` is equivalent to `self.linear_params_from_slr(self.slr)`
        and all outputs are accessible

        Args:
            mean: mean, R^n
            cov: covaraiance, R^(n x n)
            z_bar: mapped mean E(z = fn(x)) R^n
            psi: Cov(x, z) R^(n x m)
            phi: Cov(z, z) R^(m x m)
        """

        A = psi.T @ np.linalg.inv(cov)
        b = z_bar - A @ mean
        Sigma = phi - A @ cov @ A.T
        return A, b, Sigma

    @abstractmethod
    def slr(self, fn, mean, cov):
        pass

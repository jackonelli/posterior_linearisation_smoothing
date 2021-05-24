"""Statistical linear regression (SLR) with sigma points"""
from abc import ABC, abstractmethod
from functools import partial
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
        """Compute SLR quantities z_bar, psi, phi."""
        pass

    @abstractmethod
    def calc_z_bar(self, fn, mean, cov):
        """Compute SLR quantity z_bar

        Semantically identical to calling
        `z_bar, _, _ = self.slr(fn, mean, cov)`
        but more efficient in practice since it avoids computing the covariances.
        """
        pass


class SlrCache:
    def __init__(self, motion_fn, meas_fn, slr_method):
        self._motion_fn = motion_fn
        self._meas_fn = meas_fn
        self._slr = slr_method
        self.proc_lin = None
        self.meas_lin = None
        self.proc_bar = None
        self.meas_bar = None

    def update(self, means, covs):
        # TODO: single calc of sigma points.
        proc_slr = [
            self._slr.slr(partial(self._motion_fn, time_step=k), mean_k, cov_k)
            for (k, (mean_k, cov_k)) in enumerate(zip(means, covs), 1)
        ]
        self.proc_lin = [
            self._slr.linear_params_from_slr(mean_k, cov_k, *slr_) for mean_k, cov_k, slr_ in zip(means, covs, proc_slr)
        ]
        self.proc_bar = np.array([z_bar for z_bar, _, _ in proc_slr])

        meas_slr = [
            self._slr.slr(partial(self._meas_fn, time_step=k), mean_k, cov_k)
            for (k, (mean_k, cov_k)) in enumerate(zip(means, covs), 1)
        ]
        self.meas_lin = [
            self._slr.linear_params_from_slr(mean_k, cov_k, *slr_) for mean_k, cov_k, slr_ in zip(means, covs, meas_slr)
        ]
        self.meas_bar = [z_bar for z_bar, _, _ in meas_slr]

    def bars(self):
        return (self.proc_bar, self.meas_bar)

    def error_covs(self):
        return ([cov_k for (_, _, cov_k) in self.proc_lin], [cov_k for (_, _, cov_k) in self.meas_lin])

    def is_initialized(self):
        # TODO: Full
        return self.proc_lin is not None

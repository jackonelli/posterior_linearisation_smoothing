"""Stochastic linear regression (SLR)"""
import numpy as np
from post_lin_smooth.slr.distributions import Prior, Conditional
from post_lin_smooth.analytics import pos_def_check
from post_lin_smooth.linearizer import Linearizer


class Slr(Linearizer):
    """SLR"""
    def __init__(self, p_x: Prior, p_z_given_x: Conditional, num_samples: int):
        self.p_x = p_x
        self.p_z_given_x = p_z_given_x
        self.num_samples = num_samples

    def linear_params(self, mean, cov):
        """Estimate linear parameters"""
        x_sample, z_sample = self._sample(mean, cov)
        z_bar = self._z_bar(z_sample)
        psi = self._psi(x_sample, mean, z_sample, z_bar)
        phi = self._phi(z_sample, z_bar)

        A = psi.T @ np.linalg.inv(cov)
        b = z_bar - A @ _bar(x_sample)
        Sigma = phi - A @ cov @ A.T
        if not pos_def_check(Sigma, disabled=True):
            print(np.linalg.eigvals(Sigma))
            print(Sigma)
            raise ValueError("Sigma not pos def")
        return A, b, Sigma

    def _sample(self, mean, cov):
        print("Sampling x ~ p(x)")
        x_sample = self.p_x.sample(mean, cov, self.num_samples)
        print("Sampling x|z ~ p(z|x)")
        z_sample = self.p_z_given_x.sample(x_sample)
        return (x_sample, z_sample)

    @staticmethod
    def _z_bar(z_sample):
        """Calc z_bar

        Args:
            z_sample (N, D_z)

        Returns:
            z_bar (D_z,)
        """
        return _bar(z_sample)

    def _psi(self, x_sample, x_bar, z_sample, z_bar):
        """Calc Psi = Cov[x, z]
        Vectorization:
        x_diff.T @ z_diff is a matrix mult with dim's:
        (D_x, N) * (N, D_z): The sum of the product of
        each element in x_i and y_i will be computed.

        Args:
            x_sample (N, D_x)
            z_sample (N, D_z)
            z_bar (D_z,)

        Returns:
            Psi (D_x, D_z)
        """
        sample_size = x_sample.shape[0]
        x_diff = x_sample - x_bar
        z_diff = z_sample - z_bar
        cov = (x_diff.T @ z_diff)
        return cov / sample_size

    def _phi(self, z_sample, z_bar):
        """Calc Phi = Cov[z, z]
        Vectorization:
        z_diff.T @ z_diff is a matrix mult with dim's:
        (D_z, N) * (N, D_z): The sum of the product of
        each element in x_i and y_i will be computed.

        Args:
            z_sample (N, D_z)
            z_bar (D_z,)

        Returns:
            Psi (D_z, D_z)
        """
        sample_size = z_sample.shape[0]
        z_diff = z_sample - z_bar
        return z_diff.T @ z_diff / sample_size


def _bar(sample):
    return np.mean(sample, 0)

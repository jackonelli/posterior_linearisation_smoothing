"""Distribution interfaces for SLR"""
from abc import ABC, abstractmethod
import logging
import numpy as np
from scipy.stats import multivariate_normal as mvn
from src.analytics import calc_subspace_proj_matrix


class Prior(ABC):
    """Prior distribution p(x)
    This prior should in principle be a Gaussian
    but some modifications might be necessary to fulfill
    constraints in the approximated process.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, x_bar, P, num_samples):
        pass


class Conditional(ABC):
    """Conditional distribution p(z | x)"""

    @abstractmethod
    def sample(self, x_sample, num_samples: int):
        pass


class Gaussian(Prior):
    """Gaussian distribution"""

    def __init__(self):
        self._log = logging.getLogger(self.__class__.__name__)

    def sample(self, x_bar, P, num_samples):
        return mvn.rvs(mean=x_bar, cov=P, size=num_samples)


class ProjectedTruncGauss(Prior):
    """Projected truncated Gauss
    Given a mean and a covariance matrix,
    samples from a subspace where the sum over the components is constant:
    sum_c x^(i)_c = sum_c mean_c, for all i.

    NOTE: The mean is normalized
    """

    def __init__(self, num_dims):
        self.U_orth = calc_subspace_proj_matrix(num_dims)
        self._distr = TruncGauss()

    def sample(self, mean, cov, num_samples):
        mean /= mean.sum()
        proj_cov = self.U_orth @ cov @ self.U_orth
        return self._distr.sample(num_samples, mean, proj_cov)


class TruncGauss(Prior):
    """Naive truncated gaussian distr

    Samples from an ordinary `Gaussian`
    but discards samples with any negative component.
    Samples until `num_samples` ok samples are found.
    Note: the mean is also truncated to have mean_c <- el. wise max(mean_c, 0)
    """

    def __init__(self):
        self._distr = Gaussian()

    def sample(self, num_samples, mean, cov):
        mean *= mean > 0
        successful_samples = 0
        D_x = mean.shape[0]
        sample = np.empty((num_samples, D_x))
        while successful_samples < num_samples:
            candidate = self._distr.sample(mean, cov, 1)
            if (candidate > 0).all():
                sample[successful_samples, :] = candidate
                successful_samples += 1
        return sample / sample.sum(1, keepdims=True)

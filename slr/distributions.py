"""Distribution interfaces for SLR"""
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import multivariate_normal as mvn
from post_lin_smooth.analytics import pos_def_check


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
        pass

    def sample(self, x_bar, P, num_samples):
        return mvn.rvs(mean=x_bar, cov=P, size=num_samples)

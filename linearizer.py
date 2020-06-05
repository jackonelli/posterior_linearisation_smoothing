"""Linearizer interface

The KF relies on some form of linearization of
motion and measurement models, e.g.:

    - Identity, when models are linear (affine) to begin with
    - Analytical, such as the EKF
    - Sigma point methods
    - SLR
    - SLR with some analytical moments.

They commonly take an estimate of state mean and cov
and return linear (affine) parameters (A, b, Sigma)
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any


class Linearizer(ABC):
    """Linearizer interface"""
    @abstractmethod
    def linear_params(self, mean, cov):
        """Calc. linear parameters

        Args:
            mean (D_x, )
            cov (D_x, D_x)

        Returns:
            (A, b, Sigma): Linear model:
                x' = Ax + b + eps, eps ~ N(0, Sigma)
        """
        pass


class Identity(Linearizer):
    """Identity linearizer
    Trivial linearizer for linear (affine) motion and meas models
    """
    def __init__(self, A, b, Sigma):
        self.A = A
        self.b = b
        self.Sigma = Sigma

    def linear_params(self, _mean, _cov):
        return (self.A, self.b, self.Sigma)

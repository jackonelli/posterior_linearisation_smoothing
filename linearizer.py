"""Linearizer interface

The KF relies on some form of linearization of
motion and measurement models, e.g.:

    - Unit, when models are linear (affine) to begin with
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

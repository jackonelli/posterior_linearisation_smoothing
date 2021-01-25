"""Statistical linear regression (SLR) with sigma points"""
from abc import ABC, abstractmethod
import numpy as np


class Slr(ABC):
    @abstractmethod
    def linear_params(self, fn, mean, cov):
        pass

    @abstractmethod
    def slr(self, fn, mean, cov):
        pass

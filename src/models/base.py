"""Base classes for models"""
from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    @abstractmethod
    def mapping(self, state: np.ndarray) -> np.ndarray:
        """Vector to vector mapping
        For motion models, f: R^d_x --> R^d_x
        For meas. models, h: R^d_x --> R^d_y
        Must be able to handle to process a set of N vectors represented
        as a matrix R^(N, d_x)
        """

    def map_set(self, states: np.ndarray) -> np.ndarray:
        """Map multiple states
        Efficient mapping of multiple states

        vecs: Set of N states with dim. d_x, represented as a matrix R^(N, d_x)
        """
        return np.apply_along_axis(self.mapping, axis=1, arr=states)

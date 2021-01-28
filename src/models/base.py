"""Base classes for models"""
from typing import Optional
from abc import ABC, abstractmethod
from functools import partial
import numpy as np


class Model(ABC):
    @abstractmethod
    def mapping(self, state: np.ndarray, time_step: int) -> np.ndarray:
        """Vector to vector mapping
        For motion models, f: R^D_x --> R^D_x
        For meas. models, h: R^D_x --> R^D_y
        Must be able to handle to process a set of N vectors represented
        as a matrix R^(N, D_x)
        """
        pass

    def map_set(self, states: np.ndarray, time_step: Optional[int]) -> np.ndarray:
        """Map multiple states
        Efficient mapping of multiple states

        vecs: Set of N states with dim. D_x, represented as a matrix R^(N, D_x)
        """
        map_ = partial(self.mapping, time_step=time_step)
        return np.apply_along_axis(map_, axis=1, arr=states)


class Differentiable(ABC):
    @abstractmethod
    def jacobian(self, state, time_step: Optional[int]) -> np.ndarray:
        pass


class MeasModel(Model):
    @abstractmethod
    def meas_noise(self, timestep: int) -> np.ndarray:
        pass


class MotionModel(Model):
    @abstractmethod
    def proc_noise(self, timestep: int) -> np.ndarray:
        pass

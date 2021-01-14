"""Stochastic coordinated turn motion model"""
import numpy as np
from src.models.base import MotionModel, Differentiable


class NonStationaryGrowth(MotionModel, Differentiable):
    """
    state is
        x_k = actual state at time k
        k = state index, i.e. time_step
    """

    def __init__(self, a: float, b: float, c: float, proc_noise):
        self.a = a
        self.b = b
        self.c = c
        self._proc_noise = proc_noise

    def mapping(self, state):
        x_k, time_step = state
        term_1 = self.a * state
        term_2 = self.b * state / (1 + state ** 2)
        term_3 = self.c * np.cos(self.d * time_step)
        return (term_1 + term_2 + term_3, time_step + 1)

    def proc_noise(self, _time_step):
        return self._proc_noise

    def jacobian(self, state):
        raise NotImplementedError

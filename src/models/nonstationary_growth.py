"""Non-stationary growth motion model"""
import numpy as np
from src.models.base import MotionModel, Differentiable


class NonStationaryGrowth(MotionModel, Differentiable):
    """
    state is
        x_k = actual state at time step k
        k = state index, i.e. time step
    """

    def __init__(self, alpha: float, beta: float, gamma: float, delta: float, proc_noise):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self._proc_noise = proc_noise

    def mapping(self, state):
        x_k, time_step = state
        term_1 = self.alpha * x_k
        term_2 = self.beta * x_k / (1 + x_k ** 2)
        term_3 = self.gamma * np.cos(self.delta * time_step)
        return (term_1 + term_2 + term_3, time_step + 1)

    def proc_noise(self, _time_step):
        return self._proc_noise

    def jacobian(self, state):
        raise NotImplementedError

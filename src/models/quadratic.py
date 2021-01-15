"""Quadratic meas model"""
import numpy as np
from src.models.base import MotionModel, Differentiable


class Quadratic(MotionModel, Differentiable):
    """
    state is
        x_k = actual state at time step k
    """

    def __init__(self, inv_coeff: float, proc_noise):
        # Rename? 'scale' perhaps
        self.inv_coeff = inv_coeff
        self._proc_noise = proc_noise

    def mapping(self, state):
        return state ** 2 / self.inv_coeff

    def proc_noise(self, _time_step):
        return self._proc_noise

    def jacobian(self, state):
        return 2 * state / self.inv_coeff

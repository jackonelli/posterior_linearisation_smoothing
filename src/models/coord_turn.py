"""Stochastic coordinated turn motion model"""
import numpy as np
from src.models.base import MotionModel


class CoordTurn(MotionModel):
    """
    state is
        x_k = [
            pos_x,
            pos_y,
            v: speed,
            phi: angle,
            omega: angular_vel,
        ]
    """

    def __init__(self, sampling_period, proc_noise):
        self.sampling_period = sampling_period
        self._proc_noise = proc_noise

    def mapping(self, state):
        v = state[2]
        phi = state[3]
        omega = state[4]
        delta = np.array(
            [
                self.sampling_period * v * np.cos(phi),
                self.sampling_period * v * np.sin(phi),
                0,
                self.sampling_period * omega,
                0,
            ]
        )
        return state + delta

    def proc_noise(self, _k):
        return self._proc_noise

    def jacobian(self, state):
        pass

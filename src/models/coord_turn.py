"""Stochastic coordinated turn motion model"""
import numpy as np
from src.models.base import Model


class CoordTurn(Model):
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

    def __init__(self, sampling_period, process_noise):
        self.sampling_period = sampling_period
        self.process_noise = process_noise

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
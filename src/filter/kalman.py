"""Kalman filter"""
from src.filter.base import Filter


class KalmanFilter(Filter):
    """Kalman filter"""

    def __init__(self, motion_model, meas_model):
        self.A, self.b, self.Q = motion_model
        self.H, self.c, self.R = meas_model

    def _motion_lin(self, _state, _cov, _time_step):
        return (self.A, self.b, 0)

    def _meas_lin(self, _state, _cov, _time_step):
        return (self.H, self.c, 0)

    def _process_noise(self):
        return self.Q

    def _meas_noise(self):
        return self.R

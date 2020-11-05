"""Kalman filter"""
from src.filter.base import Filter


class KalmanFilter(Filter):
    """Kalman filter"""

    def __init__(self, motion_model, meas_model):
        self.motion_model = motion_model
        self.meas_model = meas_model

    def _motion_lin(self, state, cov):
        return self.motion_model

    def _meas_lin(self, state, cov):
        return self.meas_model

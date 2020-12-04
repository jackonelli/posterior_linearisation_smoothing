"""Rauch-Tung-Striebel Smoother"""
from src.smoother.base import Smoother
from src.filter.kalman import KalmanFilter


class RtsSmoother(Smoother):
    def __init__(self, motion_model, meas_model):
        super().__init__()
        self.motion_model = motion_model
        self.meas_model = meas_model

    def _motion_lin(self, _state, _cov, _time_step):
        return self.motion_model

    def _filter_seq(self, measurements, x_0_0, P_0_0):
        return KalmanFilter(self.motion_model, self.meas_model).filter_seq(measurements, x_0_0, P_0_0)

"""Kalman filter"""
from src.filter.base import Filter
from src.models.affine import AffineModel


class KalmanFilter(Filter):
    """Kalman filter"""

    def __init__(self, motion_model: AffineModel, meas_model: AffineModel):
        self._motion_model = motion_model
        self._meas_model = meas_model

    def _motion_lin(self, _state, _cov, _time_step):
        return (self._motion_model.linear_map, self._motion_model.offset, 0)

    def _meas_lin(self, _state, _cov, _time_step):
        return (self._meas_model.linear_map, self._meas_model.offset, 0)

    def _proc_noise(self, time_step):
        return self._motion_model.proc_noise(time_step)

    def _meas_noise(self, time_step):
        return self._meas_model.meas_noise(time_step)

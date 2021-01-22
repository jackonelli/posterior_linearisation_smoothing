"""Iterated Extended Kalman filter (IEKF)"""
from src.filter.base import Filter
from src.filter.ekf import ekf_lin


class Iekf(Filter):
    """Iterated Extended Kalman filter (IEKF)"""

    def __init__(self, motion_model, meas_model):
        self._motion_model = motion_model
        self._meas_model = meas_model
        self._current_means = None

    def _update_estimates(self, means, _covs):
        self._current_means = means.copy()

    def _motion_lin(self, _mean, _cov, time_step):
        mean = self._current_means[time_step, :]
        F, b = ekf_lin(self._motion_model, mean)
        return (F, b, 0)

    def _meas_lin(self, _mean, _cov, time_step):
        mean = self._current_means[time_step, :]
        H, c = ekf_lin(self._meas_model, mean)
        return (H, c, 0)

    def _proc_noise(self, time_step):
        return self._motion_model.proc_noise(time_step)

    def _meas_noise(self, time_step):
        return self._meas_model.meas_noise(time_step)

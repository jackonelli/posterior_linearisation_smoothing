"""Iterated Extended Kalman filter (IEKF)"""
from src.filter.base import Filter
from src.filter.ekf import ext_lin, ExtCache


class Iekf(Filter):
    """Iterated Extended Kalman filter (IEKF)"""

    def __init__(self, motion_model, meas_model):
        self._motion_model = motion_model
        self._meas_model = meas_model
        self._current_means = None
        self._cache = ExtCache(self._motion_model, self._meas_model)

    def _update_estimates(self, means, _covs, cache=None):
        self._current_means = means.copy()
        if cache is None:
            self._cache.update(means, None)
        else:
            self._cache = cache

    def _motion_lin(self, _mean, _cov, time_step):
        return self._cache.motion_lin[time_step]

    def _meas_lin(self, _mean, _cov, time_step):
        return self._cache.meas_lin[time_step]

    def _proc_noise(self, time_step):
        return self._motion_model.proc_noise(time_step)

    def _meas_noise(self, time_step):
        return self._meas_model.meas_noise(time_step)

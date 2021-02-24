"""Iterated posterior linearisation filter (IPLF)"""
from src.filter.base import Filter
from src.slr.sigma_points import SigmaPointSlr
from src.slr.base import SlrCache


class SigmaPointIplf(Filter):
    """Iterated posterior linearisation filter (IPLF)"""

    def __init__(self, motion_model, meas_model, sigma_point_method):
        self._motion_model = motion_model
        self._meas_model = meas_model
        self._slr = SigmaPointSlr(sigma_point_method)
        self._current_means = None
        self._current_covs = None
        self._cache = SlrCache(self._motion_model.map_set, self._meas_model.map_set, self._slr)

    def _update_estimates(self, means, covs):
        self._current_means = means.copy()
        self._current_covs = covs.copy()
        self._update_slr_cache()

    def _update_slr_cache(self):
        self._cache.update(self._current_means, self._current_covs)

    def _motion_lin(self, _mean, _cov, time_step):
        return self._cache.proc_lin[time_step]

    def _meas_lin(self, _mean, _cov, time_step):
        return self._cache.meas_lin[time_step]

    def _proc_noise(self, time_step):
        return self._motion_model.proc_noise(time_step)

    def _meas_noise(self, time_step):
        return self._meas_model.meas_noise(time_step)

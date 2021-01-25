"""Iterated posterior linearisation filter (IPLF)"""
from src.filter.base import Filter
from src.slr.sigma_points import SigmaPointSlr


class Iplf(Filter):
    """Iterated posterior linearisation filter (IPLF)"""

    def __init__(self, motion_model, meas_model, sigma_point_method):
        self._motion_model = motion_model
        self._meas_model = meas_model
        self._slr = SigmaPointSlr(sigma_point_method)
        self._current_estimates = None

    def _update_estimates(self, means, covs):
        self._current_estimates = (means.copy(), covs.copy())

    def _motion_lin(self, _mean, _cov, time_step):
        means, covs = self._current_estimates
        return self._slr.linear_params(self._motion_model.map_set, means[time_step, :], covs[time_step, :])

    def _meas_lin(self, _mean, _cov, time_step):
        means, covs = self._current_estimates
        return self._slr.linear_params(self._meas_model.map_set, means[time_step, :], covs[time_step, :])

    def _proc_noise(self, time_step):
        return self._motion_model.proc_noise(time_step)

    def _meas_noise(self, time_step):
        return self._meas_model.meas_noise(time_step)

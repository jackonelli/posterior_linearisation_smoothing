"""Sigma point Prior Linearisation Filter (PrLF)"""
from src.filter.base import Filter
from src.slr.sigma_points import SigmaPointSlr


class SigmaPointPrLf(Filter):
    """Sigma point Prior Linearisation Filter (PrLF)"""

    def __init__(self, motion_model, meas_model, sigma_point_method):
        self._motion_model = motion_model
        self._meas_model = meas_model
        self._slr = SigmaPointSlr(sigma_point_method)

    def _motion_lin(self, mean, cov, time_step):
        return self._slr.linear_params(self._mapping_with_time_step(self._motion_model.map_set, time_step), mean, cov)

    def _meas_lin(self, mean, cov, time_step):
        return self._slr.linear_params(self._mapping_with_time_step(self._meas_model.map_set, time_step), mean, cov)

    def _proc_noise(self, time_step):
        return self._motion_model.proc_noise(time_step)

    def _meas_noise(self, time_step):
        return self._meas_model.meas_noise(time_step)

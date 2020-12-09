"""Iterated posterior linearisation filter (IPLF)"""
from src.filter.base import Filter
from src.slr.sigma_points import SigmaPointSlr


class Iplf(Filter):
    """Iterated posterior linearisation filter (IPLF)"""

    def __init__(self, motion_model, meas_model):
        self.motion_model = motion_model
        self.meas_model = meas_model
        self._slr = SigmaPointSlr()
        self._current_estimates = None

    def _motion_lin(self, state, cov, _time_step):
        return self._slr.linear_params(self.motion_model.map_set, state, cov)

    def _meas_lin(self, state, cov, _time_step):
        return self._slr.linear_params(self.meas_model.map_set, state, cov)

    def _proc_noise(self, time_step):
        return self.motion_model.proc_noise(time_step)

    def _meas_noise(self):
        return self.meas_model.meas_noise

"""SLR filter"""
from src.filter.base import Filter
from src.slr.sigma_points import SigmaPointSlr


class SigmaPointSlrFilter(Filter):
    """Sigma point SLR filter"""

    def __init__(self, motion_model, meas_model):
        self.motion_model = motion_model
        self.meas_model = meas_model
        self._slr = SigmaPointSlr()

    def _motion_lin(self, state, cov):
        return self._slr.linear_params(self.motion_model.map_set, state, cov)

    def _meas_lin(self, state, cov):
        return self._slr.linear_params(self.meas_model.map_set, state, cov)

    def _process_noise(self):
        return self.motion_model.process_noise

    def _meas_noise(self):
        return self.meas_model.meas_noise

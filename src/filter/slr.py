"""SLR filter"""
from src.filter.base import Filter
from src.slr.sigma_points import SigmaPointSlr


class SigmaPointSlrFilter(Filter):
    """Slr filter"""

    def __init__(self, motion_model, meas_model):
        self.motion_model = motion_model
        self.meas_model = meas_model
        self._slr = SigmaPointSlr()

    def _motion_lin(self, state, cov):
        return self._slr.linear_params(self.motion_model.map, state, cov)

    def _meas_lin(self, state, cov):
        return self._slr.linear_params(self.meas_model.map, state, cov)

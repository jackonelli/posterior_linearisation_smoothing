"""Sigma point Prior linearisation smoother (PrLS)"""
from src.smoother.base import Smoother
from src.filter.slr import SigmaPointSlrFilter
from src.slr.sigma_points import SigmaPointSlr


class SigmaPointPrLs(Smoother):
    """Sigma point PrLs smoother"""

    def __init__(self, motion_model, meas_model):
        super().__init__()
        self._motion_model = motion_model
        self._meas_model = meas_model
        self._slr = SigmaPointSlr()

    def _motion_lin(self, mean, cov, _time_step):
        return self._slr.linear_params(self._motion_model.map_set, mean, cov)

    def _filter_seq(self, measurements, x_0_0, P_0_0):
        return SigmaPointSlrFilter(self._motion_model, self._meas_model).filter_seq(measurements, x_0_0, P_0_0)

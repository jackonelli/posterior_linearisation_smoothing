"""SLR Smoother"""
from src.smoother.base import Smoother
from src.filter.slr import SigmaPointSlrFilter
from src.slr.sigma_points import SigmaPointSlr


class SigmaPointSlrSmoother(Smoother):
    """Sigma point SLR smoother"""

    def __init__(self, motion_model, meas_model):
        self.motion_model = motion_model
        self.meas_model = meas_model
        self._slr = SigmaPointSlr()

    def _motion_lin(self, state, cov, _time_step):
        return self._slr.linear_params(self.motion_model.map_set, state, cov)

    def _filter_seq(self, measurements, x_0_0, P_0_0):
        return SigmaPointSlrFilter(self.motion_model, self.meas_model).filter_seq(measurements, x_0_0, P_0_0)

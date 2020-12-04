"""Iterated posterior linearisation filter"""
from src.smoother.base import Smoother
from src.slr.sigma_points import SigmaPointSlr


class Ipls(Smoother):
    """Iterated posterior linearisation filter"""

    def __init__(self, motion_model, meas_model):
        self.motion_model = motion_model
        self.meas_model = meas_model
        self._slr = SigmaPointSlr()

    def _motion_lin(self, state, cov):
        return self._slr.linear_params(self.motion_model.map_set, state, cov)

    def filter_and_smooth(self, measurements, x_0_0, P_0_0):
        """Overrides the base class default implementation"""

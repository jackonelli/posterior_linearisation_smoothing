"""SLR Smoother"""
from src.smoother.base import Smoother
from src.slr.sigma_points import SigmaPointSlr


class SigmaPointSlrSmoother(Smoother):
    """Slr smoother"""

    def __init__(self, motion_model):
        self.motion_model = motion_model
        self._slr = SigmaPointSlr()

    def _motion_lin(self, state, cov):
        return self._slr.linear_params(self.motion_model.map_set, state, cov)

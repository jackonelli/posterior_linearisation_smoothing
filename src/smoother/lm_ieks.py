"""Levenberg-Marquardt Iterated extended Kalman smoother (LM-IEKS)"""
from src.smoother.base import Smoother
from src.slr.sigma_points import SigmaPointSlr


class LmIeks(Smoother):
    """Levenberg-Marquardt Iterated extended Kalman smoother (LM-IEKS)"""

    def __init__(self, motion_model, meas_model):
        self.motion_model = motion_model
        self.meas_model = meas_model
        self._slr = SigmaPointSlr()

    def _motion_lin(self, state, cov):
        return self._slr.linear_params(self.motion_model.map_set, state, cov)

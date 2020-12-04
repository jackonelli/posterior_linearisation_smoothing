"""Iterated posterior linearisation filter"""
from src.smoother.base import Smoother
from src.smoother.slr import SigmaPointSlrSmoother
from src.slr.sigma_points import SigmaPointSlr


class Ipls(Smoother):
    """Iterated posterior linearisation filter"""

    def __init__(self, motion_model, meas_model):
        self.motion_model = motion_model
        self.meas_model = meas_model
        self._slr = SigmaPointSlr()
        self._current_estimates = None

    def _motion_lin(self, _state, _cov, time_step):
        means, covs = self._current_estimates()
        return self._slr.linear_params(self.motion_model.map_set, means[time_step], covs[time_step])

    def filter_and_smooth(self, measurements, x_0_0, P_0_0):
        """Overrides (extends) the base class default implementation"""
        initial_smooth_means, initial_smooth_covs, _, _ = self._first_iter(measurements, x_0_0, P_0_0)
        self._update_estimates(initial_smooth_means, initial_smooth_covs)

    def _first_iter(self, measurements, x_0_0, P_0_0):
        smoother = SigmaPointSlrSmoother(self.motion_model, self.meas_model)
        return smoother.filter_and_smooth(measurements, x_0_0, P_0_0)

    def _update_estimates(self, means, covs):
        self._current_estimates = (means, covs)

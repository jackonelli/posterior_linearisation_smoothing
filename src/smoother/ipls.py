"""Iterated posterior linearisation filter"""
from src.smoother.base import Smoother
from src.smoother.slr import SigmaPointSlrSmoother
from src.filter.iplf import Iplf
from src.slr.sigma_points import SigmaPointSlr
import numpy as np


class Ipls(Smoother):
    """Iterated posterior linearisation filter"""

    def __init__(self, motion_model, meas_model, num_iter):
        super().__init__()
        self._motion_model = motion_model
        self._meas_model = meas_model
        self._slr = SigmaPointSlr()
        self._current_estimates = None
        self.num_iter = num_iter

    def _motion_lin(self, _state, _cov, time_step):
        means, covs = self._current_estimates
        return self._slr.linear_params(self._motion_model.map_set, means[time_step], covs[time_step])

    def filter_and_smooth(self, measurements, x_0_0, P_0_0):
        """Overrides (extends) the base class default implementation"""
        xf, Pf, initial_xs, initial_Ps = self._first_iter(measurements, x_0_0, P_0_0)
        self._update_estimates(initial_xs, initial_Ps)

        current_xs, current_Ps = initial_xs, initial_Ps
        for iter_ in range(2, self.num_iter + 1):
            self._log.info(f"Iter: {iter_}")
            xf, Pf, current_xs, current_Ps = super().filter_and_smooth(measurements, x_0_0, P_0_0)
            self._update_estimates(current_xs, current_Ps)
        return xf, Pf, current_xs, current_Ps

    def _first_iter(self, measurements, x_0_0, P_0_0):
        self._log.info("Iter: 1")
        smoother = SigmaPointSlrSmoother(self._motion_model, self._meas_model)
        return smoother.filter_and_smooth(measurements, x_0_0, P_0_0)

    def _filter_seq(self, measurements, x_0_0, P_0_0):
        means, covs = self._current_estimates
        iplf = Iplf(self._motion_model, self._meas_model)
        iplf._update_estimates(means, covs)
        return iplf.filter_seq(measurements, x_0_0, P_0_0)

    def _update_estimates(self, means, covs):
        self._current_estimates = (means, covs)

"""Sigma point Iterated posterior linearisation smoother (IPLS)"""
from src.smoother.base import IteratedSmoother
from src.smoother.slr import SigmaPointPrLs
from src.filter.iplf import Iplf
from src.slr.sigma_points import SigmaPointSlr
import numpy as np


class SigmaPointIpls(IteratedSmoother):
    """Iterated posterior linearisation filter"""

    def __init__(self, motion_model, meas_model, num_iter):
        super().__init__()
        self._motion_model = motion_model
        self._meas_model = meas_model
        self._slr = SigmaPointSlr()
        self._current_estimates = None
        self.num_iter = num_iter

    def _motion_lin(self, _mean, _cov, time_step):
        means, covs = self._current_estimates
        return self._slr.linear_params(self._motion_model.map_set, means[time_step], covs[time_step])

    # def filter_and_smooth(self, measurements, m_1_0, P_1_0):
    #     """Overrides (extends) the base class default implementation"""
    #     mf, Pf, initial_ms, initial_Ps = self._first_iter(measurements, m_1_0, P_1_0)
    #     self._update_estimates(initial_ms, initial_Ps)

    #     current_ms, current_Ps = initial_ms, initial_Ps
    #     for iter_ in range(2, self.num_iter + 1):
    #         self._log.info(f"Iter: {iter_}")
    #         mf, Pf, current_ms, current_Ps = super().filter_and_smooth(measurements, m_1_0, P_1_0)
    #         self._update_estimates(current_ms, current_Ps)
    #     return mf, Pf, current_ms, current_Ps

    def _first_iter(self, measurements, m_1_0, P_1_0, cost_fn):
        self._log.info("Iter: 1")
        smoother = SigmaPointPrLs(self._motion_model, self._meas_model)
        return smoother.filter_and_smooth(measurements, m_1_0, P_1_0, cost_fn)

    def _filter_seq(self, measurements, m_1_0, P_1_0):
        means, covs = self._current_estimates
        iplf = Iplf(self._motion_model, self._meas_model)
        iplf._update_estimates(means, covs)
        return iplf.filter_seq(measurements, m_1_0, P_1_0)

    def _update_estimates(self, means, covs):
        self._current_estimates = (means.copy(), covs.copy())

"""Iterated Extended Kalman Smoother (IEKS)"""
import numpy as np
from src.smoother.base import Smoother
from src.smoother.ext.eks import Eks
from src.smoother.ext.cost import cost
from src.filter.ekf import ekf_lin
from src.filter.iekf import Iekf, ekf_lin


class Ieks(Smoother):
    """Iterated Extended Kalman Smoother (IEKS)"""

    def __init__(self, motion_model, meas_model, num_iter):
        super().__init__()
        self._motion_model = motion_model
        self._meas_model = meas_model
        self._current_means = None
        self.num_iter = num_iter

    def _motion_lin(self, _mean, _cov, time_step):
        mean = self._current_means[time_step, :]
        F, b = ekf_lin(self._motion_model, mean)
        return (F, b, 0)

    def filter_and_smooth(self, measurements, m_1_0, P_1_0):
        """Overrides (extends) the base class default implementation"""

        mf, Pf, current_ms, current_Ps = self._first_iter(measurements, m_1_0, P_1_0)
        if self.num_iter > 1:
            return self.filter_and_smooth_with_init_traj(measurements, m_1_0, P_1_0, current_ms, 2)
        else:
            return mf, Pf, current_ms, current_Ps

    def _first_iter(self, measurements, m_1_0, P_1_0):
        self._log.info("Iter: 1")
        smoother = Eks(self._motion_model, self._meas_model)
        return smoother.filter_and_smooth(measurements, m_1_0, P_1_0)

    def _run_iters(self, measurements, m_1_0, P_1_0, start_iter):
        for iter_ in range(start_iter, self.num_iter + 1):
            self._log.info(f"Iter: {iter_}")
            mf, Pf, current_ms, Ps = super().filter_and_smooth(measurements, m_1_0, P_1_0)
            self._update_estimates(current_ms)
        return mf, Pf, current_ms, Ps

    def filter_and_smooth_with_init_traj(self, measurements, m_1_0, P_1_0, init_traj, start_iter):
        """Filter and smoothing given an initial trajectory"""
        current_ms = init_traj
        self._update_estimates(current_ms)
        for iter_ in range(start_iter, self.num_iter + 1):
            self._log.info(f"Iter: {iter_}")
            mf, Pf, current_ms, current_Ps = super().filter_and_smooth(measurements, m_1_0, P_1_0)
            self._update_estimates(current_ms)
            # _cost = cost(current_ms, measurements, m_1_0, P_1_0, self._motion_model, self._meas_model)
        return mf, Pf, current_ms, current_Ps

    def _filter_seq(self, measurements, m_1_0, P_1_0):
        iekf = Iekf(self._motion_model, self._meas_model)
        iekf._update_estimates(self._current_means)
        return iekf.filter_seq(measurements, m_1_0, P_1_0)

    def _update_estimates(self, means):
        self._current_means = means.copy()

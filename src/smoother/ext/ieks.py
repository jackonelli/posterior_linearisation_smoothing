"""Iterated Extended Kalman Smoother (IEKS)"""
import numpy as np
from src.smoother.base import IteratedSmoother
from src.smoother.ext.eks import Eks
from src.filter.ekf import ekf_lin
from src.filter.iekf import Iekf, ekf_lin


class Ieks(IteratedSmoother):
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

    def _first_iter(self, measurements, m_1_0, P_1_0):
        self._log.info("Iter: 1")
        smoother = Eks(self._motion_model, self._meas_model)
        return smoother.filter_and_smooth(measurements, m_1_0, P_1_0)

    def _filter_seq(self, measurements, m_1_0, P_1_0):
        iekf = Iekf(self._motion_model, self._meas_model)
        iekf._update_estimates(self._current_means, None)
        return iekf.filter_seq(measurements, m_1_0, P_1_0)

    def _update_estimates(self, means, _covs):
        self._current_means = means.copy()

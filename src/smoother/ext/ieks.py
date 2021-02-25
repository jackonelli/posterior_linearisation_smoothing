"""Iterated Extended Kalman Smoother (IEKS)"""
from src.smoother.base import IteratedSmoother
from src.smoother.ext.eks import Eks
from src.filter.ekf import ext_lin, ExtCache
from src.filter.iekf import Iekf


class Ieks(IteratedSmoother):
    """Iterated Extended Kalman Smoother (IEKS)"""

    def __init__(self, motion_model, meas_model, num_iter):
        super().__init__()
        self._motion_model = motion_model
        self._meas_model = meas_model
        self.num_iter = num_iter
        self._cache = ExtCache(self._motion_model, self._meas_model)

    def _motion_lin(self, _mean, _cov, time_step):
        return self._cache.motion_lin[time_step]

    def _first_iter(self, measurements, m_1_0, P_1_0, cost_fn):
        smoother = Eks(self._motion_model, self._meas_model)
        return smoother.filter_and_smooth(measurements, m_1_0, P_1_0, cost_fn)

    def _filter_seq(self, measurements, m_1_0, P_1_0):
        iekf = Iekf(self._motion_model, self._meas_model)
        iekf._update_estimates(self._current_means, self._current_covs)
        return iekf.filter_seq(measurements, m_1_0, P_1_0)

    def _update_estimates(self, means, covs):
        """The 'previous estimates' which are used in the current iteration are stored in the smoother instance.
        They should only be modified through this method.
        """
        super()._update_estimates(means, covs)
        self._cache.update(means, None)

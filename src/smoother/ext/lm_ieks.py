"""Iterated Extended Kalman Smoother (IEKS)"""
import numpy as np
from src.smoother.base import Smoother
from src.smoother.eks import Eks
from src.filter.ekf import ekf_lin
from src.filter.iekf import Iekf, ekf_lin


class LmIeks(Smoother):
    """Levenberg-Marquardt Iterated Extended Kalman Smoother (LM-IEKS)"""

    def __init__(self, motion_model, meas_model, num_iter):
        super().__init__()
        self._lambda = 1e-2
        self._nu = 10
        self._motion_model = motion_model
        self._meas_model = meas_model
        self._current_means = None
        self.num_iter = num_iter
        self._store_cost_fn = list()

    def _motion_lin(self, mean, _cov, time_step):
        mean = self._current_means[time_step, :]
        F, b = ekf_lin(self._motion_model, mean)
        return (F, b, 0)

    def filter_and_smooth(self, measurements, m_1_0, P_1_0):
        """Overrides (extends) the base class default implementation"""
        _, _, initial_ms, initial_Ps = self._first_iter(measurements, m_1_0, P_1_0)
        self._update_estimates(initial_ms)
        cost_fn = _CostFn(m_1_0, P_1_0, self._motion_model, self._meas_model)
        self._store_cost_fn.append(cost_fn.cost(initial_ms, initial_Ps, measurements))

        current_ms, current_Ps = initial_ms, initial_Ps
        for iter_ in range(2, self.num_iter):
            self._log.info(f"Iter: {iter_}")
            mf, Pf, current_ms, current_Ps = super().filter_and_smooth(measurements, current_ms[0], current_Ps[0])
            new_cost = cost_fn.cost(current_ms, current_Ps, measurements)
            if new_cost < self._store_cost_fn[-1]:
                self._store_cost_fn.append(new_cost)
                self._lambda /= self._nu
                self._update_estimates(current_ms)
            else:
                self._lambda *= self._nu
        return current_ms, current_Ps, mf, Pf

    def _first_iter(self, measurements, m_1_0, P_1_0):
        self._log.info("Iter: 1")
        smoother = Eks(self._motion_model, self._meas_model)
        return smoother.filter_and_smooth(measurements, m_1_0, P_1_0)

    def _filter_seq(self, measurements, m_1_0, P_1_0):
        means = self._current_means
        lm_iekf = _LmIekf(self._motion_model, self._meas_model, self._lambda)
        lm_iekf._update_estimates(means)
        return lm_iekf.filter_seq(measurements, m_1_0, P_1_0)

    def _update_estimates(self, means):
        self._current_means = means.copy()


class _LmIekf(Iekf):
    def __init__(self, motion_model, meas_model, lambda_):
        super().__init__(motion_model, meas_model)
        self._lambda = lambda_
        self._current_means = None

    def _update_estimates(self, means):
        self._current_means = means

    def _update(self, y_k, m_k_kminus1, P_k_kminus1, R, linearization, time_step):
        """Filter update step
        Overrides (extends) the ordinary KF update with an extra pseudo-measurement of the previous state
        See base class for full docs
        """
        D_x = m_k_kminus1.shape[0]
        m_k_k, P_k_k = super()._update(y_k, m_k_kminus1, P_k_kminus1, R, linearization, time_step)
        S = P_k_k + 1 / self._lambda * np.eye(D_x)
        K = P_k_k @ np.linalg.inv(S)
        m_k_K = self._current_means[time_step, :]
        m_k_k = m_k_k + (K @ (m_k_K - m_k_k)).reshape(m_k_k.shape)
        P_k_k = P_k_k - K @ S @ K.T

        return m_k_k, P_k_k
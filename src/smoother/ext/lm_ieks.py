"""Levenberg-Marquardt Iterated Extended Kalman Smoother (LM-IEKS)"""
import numpy as np
from src.smoother.ext.eks import Eks
from src.smoother.base import IteratedSmoother
from src.filter.ekf import ExtCache
from src.filter.iekf import Iekf


class LmIeks(IteratedSmoother):
    """Levenberg-Marquardt Iterated Extended Kalman Smoother (LM-IEKS)"""

    def __init__(self, motion_model, meas_model, num_iter, cost_improv_iter_lim, lambda_, nu):
        super().__init__()
        self._motion_model = motion_model
        self._meas_model = meas_model
        self.num_iter = num_iter
        self._cost_improv_iter_lim = cost_improv_iter_lim
        self._lambda = lambda_
        self._nu = nu
        self._cache = ExtCache(self._motion_model, self._meas_model)

    def _motion_lin(self, _mean, _cov, time_step):
        return self._cache.motion_lin[time_step - 1]

    # TODO: This should also have inner LM check
    def _first_iter(self, measurements, m_1_0, P_1_0, cost_fn):
        smoother = Eks(self._motion_model, self._meas_model)
        return smoother.filter_and_smooth(measurements, m_1_0, P_1_0, cost_fn)

    def filter_and_smooth_with_init_traj(self, measurements, m_1_0, P_1_0, init_traj, start_iter, cost_fn):
        """Filter and smoothing given an initial trajectory"""
        current_ms, current_Ps = init_traj
        self._update_estimates(current_ms, current_Ps)
        prev_cost = cost_fn(current_ms)
        cost_iter = [prev_cost]
        self._log.debug(f"Initial cost: {prev_cost}")
        for iter_ in range(start_iter, self.num_iter + 1):
            self._log.debug(f"Iter: {iter_}")
            inner_iter = 0
            has_improved = False
            while has_improved is False and inner_iter < self._cost_improv_iter_lim:
                # Note: here we want to run the base `Smoother` class method.
                # I.e. we're getting the grandparent's method.
                mf, Pf, current_ms, current_Ps, _cost = super(IteratedSmoother, self).filter_and_smooth(
                    measurements, m_1_0, P_1_0, cost_fn
                )
                self._log.debug(f"Cost: {_cost}, lambda: {self._lambda}")
                if not self._lambda > 0.0:
                    self._log.info("lambda=0, skipping cost check")
                    has_improved = True
                elif _cost < prev_cost:
                    self._lambda /= self._nu
                    has_improved = True
                else:
                    self._lambda *= self._nu
                inner_iter += 1
            if inner_iter == self._cost_improv_iter_lim - 1:
                self._log.warning(f"No cost improvement for {self._cost_improv_iter_lim} iterations")
            self._update_estimates(current_ms, current_Ps)
            prev_cost = _cost
            cost_iter.append(_cost)
            # _cost = cost(current_ms, measurements, m_1_0, P_1_0, self._motion_model, self._meas_model)
        return mf, Pf, current_ms, current_Ps, np.array(cost_iter)

    def _filter_seq(self, measurements, m_1_0, P_1_0):
        lm_iekf = _LmIekf(self._motion_model, self._meas_model, self._lambda)
        lm_iekf._update_estimates(self._current_means, self._current_covs, self._cache)
        return lm_iekf.filter_seq(measurements, m_1_0, P_1_0)

    def _update_estimates(self, means, covs):
        """The 'previous estimates' which are used in the current iteration are stored in the smoother instance.
        They should only be modified through this method.
        """
        super()._update_estimates(means, covs)
        self._cache.update(means, None)


class _LmIekf(Iekf):
    def __init__(self, motion_model, meas_model, lambda_):
        super().__init__(motion_model, meas_model)
        self._lambda = lambda_
        self._current_means = None

    def _update(self, y_k, m_k_kminus1, P_k_kminus1, R, linearization, time_step):
        """Filter update step
        Overrides (extends) the ordinary KF update with an extra pseudo-measurement of the previous state
        See base class for full docs
        """
        store_ind = time_step - 1
        m_k_k, P_k_k = super()._update(y_k, m_k_kminus1, P_k_kminus1, R, linearization, time_step)
        if self._lambda > 0.0:
            D_x = m_k_kminus1.shape[0]
            S = P_k_k + 1 / self._lambda * np.eye(D_x)
            K = P_k_k @ np.linalg.inv(S)
            m_k_K = self._current_means[store_ind, :]
            m_k_k = m_k_k + (K @ (m_k_K - m_k_k)).reshape(m_k_k.shape)
            P_k_k = P_k_k - K @ S @ K.T

        return m_k_k, P_k_k

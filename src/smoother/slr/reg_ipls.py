"""Regularised Iterated Posterior Linearisation Smoother (Reg-IPLS)"""
import numpy as np
from src.smoother.base import Smoother
from src.smoother.base import IteratedSmoother
from src.filter.prlf import SigmaPointPrLf
from src.smoother.slr.prls import SigmaPointPrLs
from src.filter.iplf import Iplf
from src.slr.sigma_points import SigmaPointSlr


class SigmaPointRegIpls(IteratedSmoother):
    """Regularised Iterated Posterior Linearisation Smoother (Reg-IPLS)"""

    def __init__(self, motion_model, meas_model, sigma_point_method, num_iter, cost_improv_iter_lim, lambda_, nu):
        super().__init__()
        self._motion_model = motion_model
        self._meas_model = meas_model
        self._slr = SigmaPointSlr(sigma_point_method)
        self._sigma_point_method = sigma_point_method
        self._current_means = None
        self._current_covs = None
        self.num_iter = num_iter
        self._cost_improv_iter_lim = cost_improv_iter_lim
        self._lambda = lambda_
        self._nu = nu

    def _motion_lin(self, _mean, _cov, time_step):
        return self._slr.linear_params(
            self._motion_model.map_set, self._current_means[time_step], self._current_covs[time_step]
        )

    # TODO: This should also have inner LM check
    def _first_iter(self, measurements, m_1_0, P_1_0, cost_fn):
        self._log.info("Iter: 1")
        smoother = SigmaPointPrLs(self._motion_model, self._meas_model, self._sigma_point_method)
        return smoother.filter_and_smooth(measurements, m_1_0, P_1_0, cost_fn)

    def filter_and_smooth_with_init_traj(self, measurements, m_1_0, P_1_0, init_traj, start_iter, cost_fn):
        """Filter and smoothing given an initial trajectory"""
        current_ms, current_Ps = init_traj
        self._update_estimates(current_ms, current_Ps)
        prev_cost = cost_fn(init_traj)
        cost_iter = [prev_cost]
        self._log.debug(f"Initial cost: {prev_cost}")
        for iter_ in range(start_iter, self.num_iter + 1):
            self._log.info(f"Iter: {iter_}")
            inner_iter = 0
            has_improved = False
            while not self._terminate_inner_loop(inner_iter):
                while has_improved is False and inner_iter < self._cost_improv_iter_lim:
                    # Note: here we want to run the base `Smoother` class method.
                    # I.e. we're getting the grandparent's method.
                    mf, Pf, current_ms, current_Ps, _cost = super(IteratedSmoother, self).filter_and_smooth(
                        measurements, m_1_0, P_1_0, cost_fn
                    )
                    self._log.debug(f"Cost: {_cost}, lambda: {self._lambda}")
                    if _cost < prev_cost:
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
        lm_iekf = _RegIplf(self._motion_model, self._meas_model, self._sigma_point_method, self._lambda)
        lm_iekf._update_estimates(self._current_means, self._current_covs)
        return lm_iekf.filter_seq(measurements, m_1_0, P_1_0)

    def _terminate_inner_loop(self, inner_iter):
        return inner_iter > 0

    def _update_estimates(self, means, covs):
        self._current_means = means.copy()
        self._current_covs = covs.copy()


class _RegIplf(Iplf):
    def __init__(self, motion_model, meas_model, sigma_point_method, lambda_):
        super().__init__(motion_model, meas_model, sigma_point_method)
        self._lambda = lambda_

    def _update(self, y_k, m_k_kminus1, P_k_kminus1, R, linearization, time_step):
        """Filter update step
        Overrides (extends) the ordinary KF update with an extra pseudo-measurement of the previous state
        See base class for full docs
        """
        m_k_k, P_k_k = super()._update(y_k, m_k_kminus1, P_k_kminus1, R, linearization, time_step)
        D_x = m_k_kminus1.shape[0]
        S = P_k_k + 1 / self._lambda * np.eye(D_x)
        K = P_k_k @ np.linalg.inv(S)
        m_k_K = self._current_means[time_step, :]
        m_k_k = m_k_k + (K @ (m_k_K - m_k_k)).reshape(m_k_k.shape)
        P_k_k = P_k_k - K @ S @ K.T

        return m_k_k, P_k_k

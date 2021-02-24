"""Levenberg-Marquardt regularised Iterated Posterior Linearisation Smoother (LM-IPLS)"""
from functools import partial
import numpy as np
from src.smoother.base import Smoother
from src.cost import slr_smoothing_cost_pre_comp
from src.slr.base import SlrCache
from src.smoother.base import IteratedSmoother
from src.filter.prlf import SigmaPointPrLf
from src.smoother.slr.prls import SigmaPointPrLs
from src.filter.iplf import SigmaPointIplf
from src.slr.sigma_points import SigmaPointSlr


class SigmaPointLmIpls(IteratedSmoother):
    """Levenberg-Marquardt regularised Iterated Posterior Linearisation Smoother (LM-IPLS)"""

    def __init__(self, motion_model, meas_model, sigma_point_method, num_iter, cost_improv_iter_lim, lambda_, nu):
        super().__init__()
        self._motion_model = motion_model
        self._meas_model = meas_model
        self._slr = SigmaPointSlr(sigma_point_method)
        self._sigma_point_method = sigma_point_method
        self.num_iter = num_iter
        self._cost_improv_iter_lim = cost_improv_iter_lim
        self._lambda = lambda_
        self._nu = nu
        self._cache = SlrCache(self._motion_model.map_set, self._meas_model.map_set, self._slr)

    def _motion_lin(self, _mean, _cov, time_step):
        return self._cache.proc_lin[time_step]

    # TODO: This should also have inner LM check
    def _first_iter(self, measurements, m_1_0, P_1_0, cost_fn_prototype):
        smoother = SigmaPointPrLs(self._motion_model, self._meas_model, self._sigma_point_method)
        # Hack to use the generic `filter_and_smooth` method
        # The cost function prototype (variable covs) cannot be specialised until the `smooth_covs` are known
        # but the cost is calculated within the function (where it must be specialised).
        # Solution: - Use 'None' in the `filter_and_smooth` method => conforms to the base API.
        #           - Calculate the actual cost here, when the smooth_covs are available.
        filter_means, filter_covs, smooth_means, smooth_covs, _ = smoother.filter_and_smooth(
            measurements, m_1_0, P_1_0, None
        )
        self._update_estimates(smooth_means, smooth_covs)
        # Fix cost function
        cost_fn = self._specialise_cost_fn(cost_fn_prototype, self._cache)
        return filter_means, filter_covs, smooth_means, smooth_covs, cost_fn(smooth_means)

    def filter_and_smooth_with_init_traj(self, measurements, m_1_0, P_1_0, init_traj, start_iter, cost_fn_prototype):
        """Filter and smoothing given an initial trajectory"""
        current_ms, current_Ps = init_traj
        current_mf, current_Pf = init_traj
        if not self._cache._is_initialized():
            self._update_estimates(current_ms, current_Ps)
        cost_fn = self._specialise_cost_fn(cost_fn_prototype, self._cache)
        prev_cost = cost_fn(current_ms)
        cost_iter = [prev_cost]
        self._log.debug(f"Initial cost: {prev_cost}")
        for iter_ in range(start_iter, self.num_iter + 1):
            self._log.debug(f"Iter: {iter_}")
            loss_cand_no = 1
            has_improved = False

            while not self._terminate_inner_loop(loss_cand_no):
                print("checksum", self._current_means.sum() + self._current_covs.sum())
                while has_improved is False and loss_cand_no <= self._cost_improv_iter_lim:
                    # Note: here we want to run the base `Smoother` class method.
                    # I.e. we're getting the grandparent's method.
                    mf, Pf, current_ms, current_Ps, cost = super(IteratedSmoother, self).filter_and_smooth(
                        measurements, m_1_0, P_1_0, None
                    )
                    tmp_cache = SlrCache(self._motion_model.map_set, self._meas_model.map_set, self._slr)
                    tmp_cache.update(
                        current_ms,
                        self._current_covs,
                    )
                    tmp_cost_fn = self._specialise_cost_fn(cost_fn_prototype, tmp_cache)
                    cost = tmp_cost_fn(current_ms)
                    self._log.debug(f"Cost: {cost}, lambda: {self._lambda}, loss_cand_no: {loss_cand_no}")
                    if cost < prev_cost:
                        self._lambda /= self._nu
                        has_improved = True
                    else:
                        self._lambda *= self._nu
                    loss_cand_no += 1
                if loss_cand_no == self._cost_improv_iter_lim + 1:
                    self._log.info(f"No cost improvement for {self._cost_improv_iter_lim} iterations, returning")
                    print("checksum", self._current_means.sum() + self._current_covs.sum())
                    return current_mf, current_Pf, self._current_means, self._current_covs, np.array(cost_iter)
                # Only update the means, this is to faithfully optimise the current cost fn.
                self._update_means_only(current_ms, tmp_cache)
                prev_cost = cost
            # Now, both means and covs are updated.
            self._update_estimates(current_ms, current_Ps)
            # Fix cost function
            cost_fn = self._specialise_cost_fn(cost_fn_prototype, self._cache)
            current_mf, current_Pf = mf, Pf
            cost_iter.append(cost)
        return current_mf, current_Pf, current_ms, current_Ps, np.array(cost_iter)

    def _specialise_cost_fn(self, cost_fn_prototype, params):
        cache = params
        return partial(
            cost_fn_prototype,
            proc_bar=cache.proc_bar,
            meas_bar=cache.meas_bar,
            proc_cov=np.array(
                [err_cov_k + self._motion_model.proc_noise(k) for k, (_, _, err_cov_k) in enumerate(cache.proc_lin)]
            ),
            meas_cov=np.array(
                [err_cov_k + self._meas_model.meas_noise(k) for k, (_, _, err_cov_k) in enumerate(cache.meas_lin)]
            ),
        )

    def _filter_seq(self, measurements, m_1_0, P_1_0):
        lm_iplf = _LmIplf(self._motion_model, self._meas_model, self._sigma_point_method, self._lambda)
        lm_iplf._update_estimates(self._current_means, self._current_covs, self._cache)
        return lm_iplf.filter_seq(measurements, m_1_0, P_1_0)

    def _cost_fn_params(self):
        return self._current_covs

    def _terminate_inner_loop(self, loss_cand_no):
        return loss_cand_no > 1

    def _update_means_only(self, means, pre_comp_cache=None):
        self._current_means = means.copy()
        if pre_comp_cache is None:
            self._update_slr_cache()
        else:
            self._cache = pre_comp_cache

    def _update_estimates(self, means, covs):
        """The 'previous estimates' which are used in the current iteration are stored in the smoother instance.
        They should only be modified through this method.
        """
        super()._update_estimates(means, covs)
        self._update_slr_cache()

    def _update_slr_cache(self):
        self._cache.update(self._current_means, self._current_covs)


class _LmIplf(SigmaPointIplf):
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

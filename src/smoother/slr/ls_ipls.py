"""Line-search Iterated Posterior Linearisation Smoother (LS-IPLS)"""
import numpy as np
from src.smoother.ext.eks import Eks
from src.smoother.base import IteratedSmoother
from src.filter.ekf import ExtCache
from src.filter.iekf import Iekf
from functools import partial
from src.slr.base import SlrCache
from src.smoother.base import IteratedSmoother
from src.filter.prlf import SigmaPointPrLf
from src.smoother.slr.prls import SigmaPointPrLs
from src.filter.iplf import SigmaPointIplf
from src.slr.sigma_points import SigmaPointSlr
from src.cost_fn.slr import dir_der_slr_smoothing_cost


class SigmaPointLsIpls(IteratedSmoother):
    """Line-search Iterated Posterior Linearisation Smoother (LS-IPLS)"""

    def __init__(self, motion_model, meas_model, sigma_point_method, num_iter, line_search_method, num_points):
        super().__init__()
        self._motion_model = motion_model
        self._meas_model = meas_model
        self._slr = SigmaPointSlr(sigma_point_method)
        self._sigma_point_method = sigma_point_method
        self.num_iter = num_iter
        self._ls_method = line_search_method
        self._cache = SlrCache(self._motion_model, self._meas_model, self._slr)

    def _motion_lin(self, _mean, _cov, time_step):
        return self._cache.proc_lin[time_step - 1]

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
        cost_fn = self._specialise_cost_fn(cost_fn_prototype, (smooth_covs, self._cache.inv_cov()))
        cost = cost_fn(smooth_means)
        return filter_means, filter_covs, smooth_means, smooth_covs, cost

    def filter_and_smooth_with_init_traj(self, measurements, m_1_0, P_1_0, init_traj, start_iter, cost_fn_prototype):
        """Filter and smoothing given an initial trajectory"""
        current_ms, current_Ps = init_traj
        # If self.num_iter is too low to enter the iter loop
        mf, Pf = init_traj
        cost_iter = []
        if not self._is_initialised():
            self._update_estimates(current_ms, current_Ps)
        cost_fn = self._specialise_cost_fn(cost_fn_prototype, (self._current_covs, self._cache.inv_cov()))
        # Temporary fix which assumes that we use inexact linesearch, i.e. it breaks GridSearch.
        dir_der_prototype = partial(
            dir_der_slr_smoothing_cost,
            measurements=measurements,
            m_1_0=m_1_0,
            P_1_0=P_1_0,
            motion_fn=self._motion_model.map_set,
            meas_fn=self._meas_model.map_set,
            slr_method=self._slr,
        )
        dir_der = self._specialise_dir_der(dir_der_prototype, (self._current_covs, self._cache.inv_cov()))
        prev_cost = cost_fn(current_ms)
        for iter_ in range(start_iter, self.num_iter + 1):
            self._log.debug(f"Iter: {iter_}")
            cost_iter.append(prev_cost)
            num_iter_with_same_cost = 1
            while self._terminate_inner_loop(num_iter_with_same_cost):
                num_iter_with_same_cost += 1
                # Note: here we want to run the base `Smoother` class method.
                # I.e. we're getting the grandparent's method.
                mf, Pf, current_ms, current_Ps, cost = super(IteratedSmoother, self).filter_and_smooth(
                    measurements, m_1_0, P_1_0, cost_fn
                )
                line_search = self._ls_method(cost_fn, dir_der)
                ls_ms, alpha, ls_cost = line_search.search_next(self._current_means, current_ms)
                if ls_cost > cost:
                    self._log.warning(f"Line search did not decrease, defaulting to plain IPLS.")
                    self._update_means_only(current_ms, None)
                    # Stored for later update.
                    Ps = self._current_covs
                    prev_cost = cost
                else:
                    self._update_means_only(ls_ms, None)
                    # Stored for later update.
                    Ps = line_search.next_estimate(self._current_covs, current_Ps, alpha)
                    prev_cost = ls_cost
                # Update cost function with new covariances.
                self._update_estimates(self._current_means, Ps)
                cost_fn = self._specialise_cost_fn(cost_fn_prototype, (self._current_covs, self._cache.inv_cov()))
                dir_der = self._specialise_dir_der(dir_der_prototype, (self._current_covs, self._cache.inv_cov()))
            self._log.debug(f"Cost: {cost}, alpha: {alpha}")
            cost_fn = self._specialise_cost_fn(cost_fn_prototype, (self._current_covs, self._cache.inv_cov()))
        return mf, Pf, current_ms, current_Ps, np.array(cost_iter)

    def _filter_seq(self, measurements, m_1_0, P_1_0):
        iplf = SigmaPointIplf(self._motion_model, self._meas_model, self._sigma_point_method)
        iplf._update_estimates(self._current_means, self._current_covs, self._cache)
        return iplf.filter_seq(measurements, m_1_0, P_1_0)

    # traj, measurements, m_1_0, P_1_0, estimated_covs, motion_fn, meas_fn, motion_cov, meas_cov, slr_method
    def _specialise_cost_fn(self, cost_fn_prototype, params):
        (
            estimated_covs,
            (
                motion_cov_inv,
                meas_cov_inv,
            ),
        ) = params
        return partial(
            cost_fn_prototype,
            estimated_covs=estimated_covs,
            motion_cov_inv=motion_cov_inv,
            meas_cov_inv=meas_cov_inv,
        )

    def _specialise_dir_der(self, dir_der_prototype, params):
        (
            estimated_covs,
            (
                motion_cov_inv,
                meas_cov_inv,
            ),
        ) = params
        return partial(
            dir_der_prototype,
            estimated_covs=estimated_covs,
            motion_cov_inv=motion_cov_inv,
            meas_cov_inv=meas_cov_inv,
        )

    def _is_initialised(self):
        return self._cache.is_initialized() and self._current_means is not None and self._current_covs is not None

    def _cost_fn_params(self):
        return self._current_covs

    def _terminate_inner_loop(self, inner_loop_iter):
        return inner_loop_iter < 2

    def _update_means_only(self, means, pre_comp_cache=None):
        self._current_means = means.copy()
        if pre_comp_cache is None:
            self._cache.update(self._current_means, self._current_covs)
        else:
            self._cache = pre_comp_cache

    def _update_estimates(self, means, covs):
        """The 'previous estimates' which are used in the current iteration are stored in the smoother instance.
        They should only be modified through this method.
        """
        super()._update_estimates(means, covs)
        self._cache.update(self._current_means, self._current_covs)

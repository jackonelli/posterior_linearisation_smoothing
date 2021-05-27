"""Sigma point Iterated posterior linearisation smoother (IPLS)"""
from functools import partial
import numpy as np
from src.smoother.base import IteratedSmoother
from src.smoother.slr.prls import SigmaPointPrLs
from src.filter.iplf import SigmaPointIplf
from src.slr.sigma_points import SigmaPointSlr
from src.slr.base import SlrCache


class SigmaPointIpls(IteratedSmoother):
    """Iterated posterior linearisation filter"""

    def __init__(self, motion_model, meas_model, sigma_point_method, num_iter):
        super().__init__()
        self._motion_model = motion_model
        self._meas_model = meas_model
        self._slr = SigmaPointSlr(sigma_point_method)
        self._sigma_point_method = sigma_point_method
        self.num_iter = num_iter
        self._cache = SlrCache(self._motion_model, self._meas_model, self._slr)

    def _motion_lin(self, _mean, _cov, time_step):
        return self._cache.proc_lin[time_step - 1]

    def _first_iter(self, measurements, m_1_0, P_1_0, cost_fn_prototype):
        smoother = SigmaPointPrLs(self._motion_model, self._meas_model, self._sigma_point_method)
        mf, Pf, ms, Ps, _ = smoother.filter_and_smooth(measurements, m_1_0, P_1_0, None)
        self._update_estimates(ms, Ps)
        if cost_fn_prototype is not None:
            cost_fn = self._specialise_cost_fn(cost_fn_prototype, self._cost_fn_params())
            cost = cost_fn(ms)
        else:
            cost = None
        return mf, Pf, ms, Ps, cost

    def _filter_seq(self, measurements, m_1_0, P_1_0):
        iplf = SigmaPointIplf(self._motion_model, self._meas_model, self._sigma_point_method)
        iplf._update_estimates(self._current_means, self._current_covs, self._cache)
        return iplf.filter_seq(measurements, m_1_0, P_1_0)

    def _specialise_cost_fn(self, cost_fn_prototype, params):
        if cost_fn_prototype is not None:
            (
                (motion_bar, meas_bar),
                (
                    motion_cov_inv,
                    meas_cov_inv,
                ),
            ) = params
            return partial(
                cost_fn_prototype,
                motion_bar=motion_bar,
                meas_bar=meas_bar,
                motion_cov_inv=motion_cov_inv,
                meas_cov_inv=meas_cov_inv,
            )
        else:
            return None

    def _cost_fn_params(self):
        return self._cache.bars(), self._cache.error_covs()

    def _update_estimates(self, means, covs):
        """The 'previous estimates' which are used in the current iteration are stored in the smoother instance.
        They should only be modified through this method.
        """
        super()._update_estimates(means, covs)
        self._cache.update(self._current_means, self._current_covs)

    def _is_initialised(self):
        return self._cache.is_initialized() and self._current_means is not None and self._current_covs is not None

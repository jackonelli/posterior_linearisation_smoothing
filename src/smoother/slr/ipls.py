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
        self._cache = SlrCache(self._motion_model.map_set, self._meas_model.map_set, self._slr)

    def _motion_lin(self, _mean, _cov, time_step):
        return self._cache.proc_lin[time_step - 1]

    def _first_iter(self, measurements, m_1_0, P_1_0, cost_fn_prototype):
        smoother = SigmaPointPrLs(self._motion_model, self._meas_model, self._sigma_point_method)
        return smoother.filter_and_smooth(measurements, m_1_0, P_1_0, None)

    def _filter_seq(self, measurements, m_1_0, P_1_0):
        iplf = SigmaPointIplf(self._motion_model, self._meas_model, self._sigma_point_method)
        iplf._update_estimates(self._current_means, self._current_covs, self._cache)
        return iplf.filter_seq(measurements, m_1_0, P_1_0)

    def _specialise_cost_fn(self, cost_fn_prototype, params):
        (
            (proc_bar, meas_bar),
            (
                proc_lin_cov,
                meas_lin_cov,
            ),
        ) = params
        return partial(
            cost_fn_prototype,
            proc_bar=proc_bar,
            meas_bar=meas_bar,
            proc_cov=np.array(
                [err_cov_k + self._motion_model.proc_noise(k) for k, err_cov_k in enumerate(proc_lin_cov)]
            ),
            meas_cov=[err_cov_k + self._meas_model.meas_noise(k) for k, err_cov_k in enumerate(meas_lin_cov)],
        )

    def _cost_fn_params(self):
        return self._cache.bars(), self._cache.error_covs()

    def _update_estimates(self, means, covs):
        """The 'previous estimates' which are used in the current iteration are stored in the smoother instance.
        They should only be modified through this method.
        """
        super()._update_estimates(means, covs)
        self._cache.update(self._current_means, self._current_covs)

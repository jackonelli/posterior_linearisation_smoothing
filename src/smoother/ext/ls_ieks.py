"""Line-search Iterated Extended Kalman Smoother (LS-IEKS)"""
import numpy as np
from src.smoother.ext.eks import Eks
from src.smoother.base import IteratedSmoother
from src.filter.ekf import ExtCache
from src.filter.iekf import Iekf


class LsIeks(IteratedSmoother):
    """Line-search Iterated Extended Kalman Smoother (LS-IEKS)"""

    def __init__(self, motion_model, meas_model, num_iter, line_search_method):
        super().__init__()
        self._motion_model = motion_model
        self._meas_model = meas_model
        self.num_iter = num_iter
        self._ls_method = line_search_method
        self._cache = ExtCache(self._motion_model, self._meas_model)

    def _motion_lin(self, _mean, _cov, time_step):
        return self._cache.motion_lin[time_step - 1]

    # TODO: This should also have inner LS check
    def _first_iter(self, measurements, m_1_0, P_1_0, cost_fn):
        smoother = Eks(self._motion_model, self._meas_model)
        return smoother.filter_and_smooth(measurements, m_1_0, P_1_0, cost_fn)

    def filter_and_smooth_with_init_traj(self, measurements, m_1_0, P_1_0, init_traj, start_iter, cost_fn):
        """Filter and smoothing given an initial trajectory"""
        current_ms, current_Ps = init_traj
        # If self.num_iter is too low to enter the iter loop
        mf, Pf = init_traj
        self._update_estimates(current_ms, current_Ps)
        prev_cost = cost_fn(current_ms)
        cost_iter = [prev_cost]
        self._log.debug(f"Initial cost: {prev_cost}")
        for iter_ in range(start_iter, self.num_iter + 1):
            self._log.debug(f"Iter: {iter_}")
            # Note: here we want to run the base `Smoother` class method.
            # I.e. we're getting the grandparent's method.
            current_mf, current_Pf, current_ms, current_Ps, cost = super(IteratedSmoother, self).filter_and_smooth(
                measurements, m_1_0, P_1_0, cost_fn
            )
            ls_ms, alpha, ls_cost = self._ls_method.search_next(self._current_means, current_ms)
            if ls_cost > cost:
                self._log.warning(f"Line search did not decrease, defaulting to plain IEKS.")
                self._update_estimates(current_ms, current_Ps)
                prev_cost = cost
                mf = current_mf
                Pf = current_Pf
            else:
                self._update_estimates(ls_ms, current_Ps)
                prev_cost = ls_cost
                mf = mf + alpha * (current_mf - self._current_means)
            cost_iter.append(prev_cost)
            # _cost = cost(current_ms, measurements, m_1_0, P_1_0, self._motion_model, self._meas_model)
        return mf, Pf, current_ms, current_Ps, np.array(cost_iter)

    def _filter_seq(self, measurements, m_1_0, P_1_0):
        iekf = Iekf(self._motion_model, self._meas_model)
        iekf._update_estimates(self._current_means, self._current_covs, self._cache)
        return iekf.filter_seq(measurements, m_1_0, P_1_0)

    def _update_estimates(self, means, covs):
        """The 'previous estimates' which are used in the current iteration are stored in the smoother instance.
        They should only be modified through this method.
        """
        super()._update_estimates(means, covs)
        self._cache.update(means, None)

    def _is_initialised(self):
        return self._cache.is_initialized() and self._current_means is not None

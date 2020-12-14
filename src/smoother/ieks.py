"""Iterated Extended Kalman Smoother (IEKS)"""
import numpy as np
from src.smoother.base import Smoother
from src.smoother.eks import Eks
from src.filter.ekf import ekf_lin
from src.filter.iekf import Iekf, ekf_lin


class Ieks(Smoother):
    """Iterated Extended Kalman Smoother (IEKS)"""

    def __init__(self, motion_model, meas_model, num_iter):
        super().__init__()
        self._motion_model = motion_model
        self._meas_model = meas_model
        self._current_means = None
        self.num_iter = num_iter

    def _motion_lin(self, state, _cov, time_step):
        mean = self._current_means[time_step]
        F, b = ekf_lin(self._motion_model, mean)
        return (F, b, 0)

    def filter_and_smooth(self, measurements, x_1_0, P_1_0):
        """Overrides (extends) the base class default implementation"""
        # initial_xs, _, _, _ = self._first_iter(measurements, x_1_0, P_1_0)
        D_x = x_1_0.shape[0]
        K = measurements.shape[0]

        current_xs = np.zeros((K, D_x))
        self._update_estimates(current_xs)
        for iter_ in range(1, self.num_iter + 1):
            self._log.info(f"Iter: {iter_}")
            xf, Pf, current_xs, current_Ps = super().filter_and_smooth(measurements, x_1_0, P_1_0)
            self._update_estimates(current_xs)
        return xf, Pf, current_xs, current_Ps

    def _first_iter(self, measurements, x_0_0, P_0_0):
        self._log.info("Iter: 1")
        smoother = Eks(self._motion_model, self._meas_model)
        return smoother.filter_and_smooth(measurements, x_0_0, P_0_0)

    def _filter_seq(self, measurements, x_0_0, P_0_0):
        means = self._current_means
        iekf = Iekf(self._motion_model, self._meas_model)
        iekf._update_estimates(means)
        return iekf.filter_seq(measurements, x_0_0, P_0_0)

    def _update_estimates(self, means):
        self._current_means = means


class GnIeks(Smoother):
    """Gauss-Newton Iterated Extended Kalman Smoother (GN-IEKS)
    Depends on choice of line search algorithm

    TODO: Implement more sophisticated line search.
    With no line search, this algorithm is equivalent to IEKS
    """

    def __init__(self, motion_model, meas_model, num_iter):
        super().__init__()
        self._motion_model = motion_model
        self._meas_model = meas_model
        self._current_means = None
        self.num_iter = num_iter

    def _motion_lin(self, state, _cov, time_step):
        mean = self._current_means[time_step]
        F, b = ekf_lin(self._motion_model, mean)
        return (F, b, 0)

    def filter_and_smooth(self, measurements, x_0_0, P_0_0):
        """Overrides (extends) the base class default implementation"""
        initial_xs, initial_Ps, xf, Pf = self._first_iter(measurements, x_0_0, P_0_0)
        self._update_estimates(initial_xs)

        current_xs, current_Ps = initial_xs, initial_Ps
        for iter_ in range(2, self.num_iter):
            self._log.info(f"Iter: {iter_}")
            current_xs, current_Ps, xf, Pf = super().filter_and_smooth(measurements, current_xs[0], current_Ps[0])
            self._update_estimates(current_xs)
        return current_xs, current_Ps, xf, Pf

    def _first_iter(self, measurements, x_0_0, P_0_0):
        self._log.info("Iter: 1")
        smoother = Eks(self._motion_model, self._meas_model)
        return smoother.filter_and_smooth(measurements, x_0_0, P_0_0)

    def _filter_seq(self, measurements, x_0_0, P_0_0):
        means = self._current_means
        iekf = Iekf(self._motion_model, self._meas_model)
        iekf._update_estimates(means)
        return iekf.filter_seq(measurements, x_0_0, P_0_0)

    def _update_estimates(self, means):
        self._current_means = means


# TODO: GN Params data class


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

    def _motion_lin(self, state, _cov, time_step):
        mean = self._current_means[time_step]
        F, b = ekf_lin(self._motion_model, mean)
        return (F, b, 0)

    def filter_and_smooth(self, measurements, x_0_0, P_0_0):
        """Overrides (extends) the base class default implementation"""
        initial_xs, initial_Ps, xf, Pf = self._first_iter(measurements, x_0_0, P_0_0)
        self._update_estimates(initial_xs)
        cost_fn = _CostFn(x_0_0, P_0_0, self._motion_model, self._meas_model)
        self._store_cost_fn.append(cost_fn.cost(initial_xs, initial_Ps, measurements))

        current_xs, current_Ps = initial_xs, initial_Ps
        for iter_ in range(2, self.num_iter):
            self._log.info(f"Iter: {iter_}")
            current_xs, current_Ps, xf, Pf = super().filter_and_smooth(measurements, current_xs[0], current_Ps[0])
            new_cost = cost_fn.cost(current_xs, current_Ps, measurements)
            if new_cost < self._store_cost_fn[-1]:
                self._store_cost_fn.append(new_cost)
                self._lambda /= self._nu
                self._update_estimates(current_xs)
            else:
                self._lambda *= self._nu
        return current_xs, current_Ps, xf, Pf

    def _first_iter(self, measurements, x_0_0, P_0_0):
        self._log.info("Iter: 1")
        smoother = Eks(self._motion_model, self._meas_model)
        return smoother.filter_and_smooth(measurements, x_0_0, P_0_0)

    def _filter_seq(self, measurements, x_0_0, P_0_0):
        means = self._current_means
        lm_iekf = _LmIekf(self._motion_model, self._meas_model, self._lambda)
        lm_iekf._update_estimates(means)
        return lm_iekf.filter_seq(measurements, x_0_0, P_0_0)

    def _update_estimates(self, means):
        self._current_means = means


class _LmIekf(Iekf):
    def __init__(self, motion_model, meas_model, lambda_):
        super().__init__(motion_model, meas_model)
        self._lambda = lambda_
        self._current_means = None

    def _update_estimates(self, means):
        self._current_means = means

    def _update(self, y_k, x_k_kminus1, P_k_kminus1, R, linearization, time_step):
        """Filter update step
        Overrides (extends) the ordinary KF update with an extra pseudo-measurement of the previous state
        See base class for full docs
        """
        D_x = x_k_kminus1.shape[0]
        x_k_k, P_k_k = super()._update(y_k, x_k_kminus1, P_k_kminus1, R, linearization, time_step)
        S = P_k_k + 1 / self._lambda * np.eye(D_x)
        K = P_k_k @ np.linalg.inv(S)
        x_k_K = self._current_means[time_step]
        x_k_k = x_k_k + (K @ (x_k_K - x_k_k)).reshape(x_k_k.shape)
        P_k_k = P_k_k - K @ S @ K.T

        return x_k_k, P_k_k


class _CostFn:
    def __init__(self, prior_mean, prior_cov, motion_model, meas_model):
        self._x_0 = prior_mean
        self._P_0 = prior_cov
        self._motion_model = motion_model
        self._meas_model = meas_model

    def cost(self, means, covs, measurements):
        diff = means[1, :] - self._x_0
        _cost = diff.T @ self._P_0 @ diff
        # TODO: means should be in R^K, but are in R^K+1 (including zero).
        # TODO: vectorise with map sets
        for k in range(1, means.shape[0] - 1):
            diff = means[k + 1, :] - self._motion_model.mapping(means[k, :])
            _cost += diff.T @ self._motion_model.proc_noise(k) @ diff
            # measurements are zero indexed, i.e. k-1 --> y_k
            diff = measurements[k - 1, :] - self._meas_model.mapping(means[k, :])
            _cost += diff.T @ self._meas_model.meas_noise(k) @ diff
        diff = measurements[-1, :] - self._meas_model.mapping(means[-1, :])
        _cost += diff.T @ self._meas_model.meas_noise(k) @ diff

        return _cost

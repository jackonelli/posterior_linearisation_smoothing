"""Iterated Extended Kalman Smoother (IEKS)"""
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


class GnIeks(Smoother):
    """Gauss-Newton Iterated Extended Kalman Smoother (IEKS)"""

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

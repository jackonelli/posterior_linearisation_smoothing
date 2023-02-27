"""Line search method for numerical optimisation"""
from abc import abstractmethod, ABC
import numpy as np


class LineSearch(ABC):
    @abstractmethod
    def search_next(self, x_0, x_1):
        """Compute next iterate and step length alpha > 0 such that the next iterate becomes

        x_next = x_0 + alpha (x_1 - x_0)

        returns:
            x_next, alpha, cost(x_next)
        """
        pass

    @staticmethod
    def next_estimate(x_0, x_1, alpha):
        return alpha * (x_1 - x_0) + x_0


class GridSearch(LineSearch):
    def __init__(self, cost_fn, num_points):
        self._cost_fn = cost_fn
        self._num_points = num_points

    def search_next(self, x_0, x_1):
        alphas = np.linspace(0, 1, self._num_points)
        cands = np.array([self.next_estimate(x_0, x_1, alpha) for alpha in alphas])
        costs = np.array([self._cost_fn(cand) for cand in cands])
        min_ind = np.argmin(costs)
        return cands[min_ind], alphas[min_ind], costs[min_ind]


class ArmijoLineSearch(LineSearch):
    """Inexact line-search with the Armijo condition

    See chapter 3.1 "Numerical optimisation", Nocedal (2006)
    """

    def __init__(self, cost_fn, dir_der_fn, c_1, alpha_start=1.0, num_trials=10, tau=0.5):
        self._cost_fn = cost_fn
        self._dir_der_fn = dir_der_fn
        self._c_1 = c_1
        self._alpha_start = alpha_start
        self._num_trials = num_trials
        self._tau = tau

    def _suff_decrease_condition(self, x_0, x_1, alpha, dir_der_0, f_0) -> bool:
        search_dir = x_1 - x_0
        lhs = self._cost_fn(x_0 + alpha * search_dir)
        rhs = f_0 + self._c_1 * alpha * dir_der_0
        return lhs <= rhs

    def search_next(self, x_0, x_1):
        alpha = self._alpha_start
        search_dir = x_1 - x_0
        dir_der_0 = self._dir_der_fn(x_0, search_dir)
        f_0 = self._cost_fn(x_0)
        i, done = 0, False
        while not done and i < self._num_trials:
            if self._suff_decrease_condition(x_0, x_1, alpha, dir_der_0, f_0):
                done = True
            else:
                alpha *= self._tau
            i += 1
        x_next = self.next_estimate(x_0, x_1, alpha)
        return x_next, alpha, self._cost_fn(x_next)


class ArmijoWolfeLineSearch(ArmijoLineSearch):
    """Combined Armijo-Wolfe step length conditions

    See chapter 3.1 "Numerical optimisation", Nocedal (2006)
    """

    def __init__(self, cost_fn, dir_der_fn, c_1, c_2, alpha_start=1.0, num_trials=10, tau=0.5):
        super().__init__(cost_fn, dir_der_fn, c_1, alpha_start, num_trials, tau)
        self._c_2 = c_2

    def _curvature_condition(self, x_0, x_1, alpha, dir_der_0) -> bool:
        search_dir = x_1 - x_0
        x_alpha = x_0 + alpha * search_dir
        lhs = self._dir_der_fn(x_alpha, search_dir)
        rhs = self._c_2 * dir_der_0
        return lhs >= rhs

    def search_next(self, x_0, x_1):
        alpha = self._alpha_start
        search_dir = x_1 - x_0

        dir_der_0 = self._dir_der_fn(x_0, search_dir)
        f_0 = self._cost_fn(x_0)
        i, done = 0, False
        while not done and i < self._num_trials:
            if self._suff_decrease_condition(x_0, x_1, alpha, dir_der_0, f_0) and self._curvature_condition(
                x_0, x_1, alpha, dir_der_0
            ):
                done = True
            else:
                alpha *= self._tau
            i += 1
        x_next = self.next_estimate(x_0, x_1, alpha)
        return x_next, alpha, self._cost_fn(x_next)

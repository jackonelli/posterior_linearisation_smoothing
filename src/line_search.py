"""Line search method for numerical optimisation"""
from abc import abstractmethod, ABC
import numpy as np


class LineSearch(ABC):
    @abstractmethod
    def search_next(self, x_0, x_1):
        """Compute next iterate and step length alpha > 0 such that the next iterate becomes

        x_next = x_0 + alpha (x_1 - x_0)
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
    """Armijo step length condition

    See chapter 3.1 "Numerical optimisation", Nocedal (2006)
    """

    def __init__(self, cost_fn, grad_fn, c_1):
        self._cost_fn = cost_fn
        self._grad_fn = grad_fn
        self._c_1 = c_1

    def suff_decrease_condition(self, x_0, x_1, alpha) -> bool:
        search_dir = x_1 - x_0
        lhs = self._cost_fn(x_0 + alpha * x_1)
        rhs = self._cost_fn(x_0) + self._c_1 * alpha * search_dir.T @ self._grad_fn(x_0)
        return lhs <= rhs

    def search_next(self, x_0, x_1):
        alphas = np.linspace(0, 1, self._num_points)
        cands = np.array([self.next_estimate(x_0, x_1, alpha) for alpha in alphas])
        costs = np.array([self._cost_fn(cand) for cand in cands])
        min_ind = np.argmin(costs)
        return cands[min_ind], alphas[min_ind], costs[min_ind]

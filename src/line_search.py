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
        rhs = self._cost_fn(x_0) + self._c_1 * alpha * self._grad_fn(x_0) @ search_dir
        return lhs <= rhs

    def search_next(self, x_0, x_1):
        alphas = np.linspace(0, 1, self._num_points)
        cands = np.array([self.next_estimate(x_0, x_1, alpha) for alpha in alphas])
        costs = np.array([self._cost_fn(cand) for cand in cands])
        min_ind = np.argmin(costs)
        return cands[min_ind], alphas[min_ind], costs[min_ind]


def jacobian_analytical_smoothing_cost(
    traj, measurements, m_1_0, P_1_0, motion_model: MotionModel, meas_model: MeasModel
):
    """Gradient of the cost function `analytical_smoothing_cost`

    Args:
        traj: states for a time sequence 1, ..., K
            represented as a np.array(K, D_x).
            (The actual variable in the cost function)
        measurements: measurements for a time sequence 1, ..., K
            represented as a list of length K of np.array(D_y,)
    """
    K = len(measurements)
    prior_diff = traj[0, :] - m_1_0
    _cost = prior_diff.T @ np.linalg.inv(P_1_0) @ prior_diff

    motion_diff = traj[1:, :] - motion_model.map_set(traj[:-1, :], None)
    meas_diff = measurements - meas_model.map_set(traj, None)
    for k in range(0, K - 1):
        _cost += motion_diff[k, :].T @ np.linalg.inv(motion_model.proc_noise(k)) @ motion_diff[k, :]
        # measurements are zero indexed, i.e. k-1 --> y_k
        if any(np.isnan(meas_diff[k, :])):
            continue
        _cost += meas_diff[k, :].T @ np.linalg.inv(meas_model.meas_noise(k)) @ meas_diff[k, :]
    _cost += meas_diff[-1, :].T @ np.linalg.inv(meas_model.meas_noise(K)) @ meas_diff[-1, :]

    return _cost

"""Range bearing meas model"""
import numpy as np
from src.models.base import MeasModel, Differentiable
from scipy.stats import multivariate_normal as mvn


class RangeBearing(MeasModel, Differentiable):
    """ pos np.array(2,)"""

    def __init__(self, pos, meas_noise):
        self.pos = pos
        self._meas_noise = meas_noise

    def mapping(self, state, time_step=None):
        range_ = np.sqrt(np.sum((state[:2] - self.pos) ** 2))
        bearing = angle(state, self.pos)
        return np.array([range_, bearing])

    def meas_noise(self, time_step):
        return self._meas_noise

    def jacobian(self, state, time_step=None):
        x, y = state[0], state[1]
        s_x, s_y = self.pos[0], self.pos[1]
        delta_x, delta_y = x - s_x, y - s_y

        range_den = np.sqrt((delta_x) ** 2 + (delta_y) ** 2)
        range_diff = np.array([delta_x / range_den, delta_y / range_den])

        bearings_diff = atan2_jacobian(delta_x, delta_y)

        non_zero = np.row_stack((range_diff, bearings_diff))

        return np.column_stack((non_zero, np.zeros((2, 3))))


class MultiSensorRange(MeasModel, Differentiable):
    def __init__(self, sensors, meas_noise):
        """
        Num. sensors = N
        sensors (np.ndarray): (N, D_y)

        TODO: Actually extend to arb. num sensors (curr two (Jacobian))
        """
        self.sensors = sensors
        self._meas_noise = meas_noise

    def mapping(self, state, time_step=None):
        return np.apply_along_axis(lambda pos: euclid_dist(state[:2], pos), axis=1, arr=self.sensors)

    def meas_noise(self, time_step):
        return self._meas_noise

    def jacobian(self, state, time_step=None):
        zeros_len = state.shape[0] - 2
        s_1 = self.sensors[0, :]
        s_2 = self.sensors[1, :]
        s_1_den = np.sqrt((state[0] - s_1[0]) ** 2 + (state[1] - s_1[1]) ** 2)
        s_2_den = np.sqrt((state[0] - s_2[0]) ** 2 + (state[1] - s_2[1]) ** 2)
        H_11 = -(s_1[0] - state[0]) / s_1_den
        H_12 = -(s_1[1] - state[1]) / s_1_den
        H_21 = -(s_2[0] - state[0]) / s_2_den
        H_22 = -(s_2[1] - state[1]) / s_2_den
        return np.column_stack((np.array([[H_11, H_12], [H_21, H_22]]), np.zeros((2, zeros_len))))

    def sample(self, states):
        means = self.map_set(states)
        num_samples, D_y = means.shape
        noise = mvn.rvs(mean=np.zeros((D_y,)), cov=self.meas_noise(None), size=num_samples)
        return means + noise


class MultiSensorBearings(MeasModel, Differentiable):
    def __init__(self, sensors, meas_noise):
        """
        Num. sensors = N
        sensors (np.ndarray): (N, D_y)

        TODO: Actually extend to arb. num sensors (curr two (Jacobian))
        """
        self.sensors = sensors
        self._meas_noise = meas_noise

    def mapping(self, state, time_step=None):
        return np.apply_along_axis(lambda pos: angle(state[:2], pos), axis=1, arr=self.sensors)

    def meas_noise(self, time_step):
        return self._meas_noise

    def jacobian(self, state, time_step=None):
        zeros_len = state.shape[0] - 2
        s_1 = self.sensors[0, :]
        s_2 = self.sensors[1, :]
        x, y = state[0], state[1]

        dh1 = atan2_jacobian(x - s_1[0], y - s_1[1])
        dh2 = atan2_jacobian(x - s_2[0], y - s_2[1])
        return np.column_stack((np.array([dh1, dh2]), np.zeros((2, zeros_len))))

    def sample(self, states):
        means = self.map_set(states)
        num_samples, D_y = means.shape
        noise = mvn.rvs(mean=np.zeros((D_y,)), cov=self.meas_noise(None), size=num_samples)
        return means + noise


def atan2_jacobian(x, y):
    num = x ** 2 + y ** 2
    return np.array([-y / num, x / num])


def euclid_dist(p_1, p_2):
    return np.sqrt(np.sum((p_1 - p_2) ** 2))


def angle(p_1, p_2):
    delta_x = p_1[0] - p_2[0]
    delta_y = p_1[1] - p_2[1]
    return np.arctan2(delta_y, delta_x)


def to_cartesian_coords(meas, pos):
    """Maps a range and bearing measurement to cartesian coords

    Args:
        meas np.array(D_y,)
        pos np.array(2,)

    Returns:
        coords np.array(2,)
    """
    delta_x = meas[0] * np.cos(meas[1])
    delta_y = meas[0] * np.sin(meas[1])
    coords = np.array([delta_x, delta_y]) + pos
    return coords

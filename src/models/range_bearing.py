"""Range bearing meas model"""
import numpy as np
from src.models.base import MeasModel, Differentiable
from scipy.stats import multivariate_normal as mvn


class RangeBearing(MeasModel):
    """ pos np.array(2,)"""

    def __init__(self, pos, meas_noise):
        self.pos = pos
        self._meas_noise = meas_noise

    def mapping(self, state, time_step=0):
        range_ = np.sqrt(np.sum((state[:2] - self.pos) ** 2))
        bearing = np.arctan2(state[1] - self.pos[1], state[0] - self.pos[0])
        return np.array([range_, bearing])

    def meas_noise(self, time_step):
        return self._meas_noise


class MultiSensorRange(MeasModel, Differentiable):
    def __init__(self, sensors, meas_noise):
        """
        Num. sensors = N
        sensors (np.ndarray): (N, D_y)
        """
        self.sensors = sensors
        self._meas_noise = meas_noise

    def mapping(self, state, time_step=0):
        return np.apply_along_axis(lambda pos: euclid_dist(state[:2], pos), axis=1, arr=self.sensors)

    def meas_noise(self, time_step):
        return self._meas_noise

    def jacobian(self, state, time_step=0):
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


def euclid_dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


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

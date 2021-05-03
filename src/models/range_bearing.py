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
        range_ = _euclid_dist(state[:2], self.pos)
        bearing = _angle(state, self.pos)
        return np.array([range_, bearing])

    def meas_noise(self, time_step):
        return self._meas_noise

    def jacobian(self, state, time_step=None):
        x, y = state[0], state[1]
        s_x, s_y = self.pos[0], self.pos[1]
        delta_x, delta_y = x - s_x, y - s_y

        range_derivative = _euclid_dist_jacobian(delta_x, delta_y)
        bearings_derivative = _angle_jacobian(delta_x, delta_y)
        non_zero = np.row_stack((range_derivative, bearings_derivative))

        return np.column_stack((non_zero, np.zeros((2, 3))))


class MultiSensorRange(MeasModel, Differentiable):
    def __init__(self, sensors, meas_noise):
        """
        Num. sensors = N
        sensors (np.ndarray): (N, D_y)
        """
        self.sensors = sensors
        self._meas_noise = meas_noise

    def mapping(self, state, time_step=None):
        return np.apply_along_axis(lambda pos: _euclid_dist(state[:2], pos), axis=1, arr=self.sensors)

    def meas_noise(self, time_step):
        return self._meas_noise

    def jacobian(self, state, time_step=None):
        zeros_len = state.shape[0] - 2
        x, y = state[0], state[1]
        non_zero = np.apply_along_axis(
            lambda pos: _euclid_dist_jacobian(x - pos[0], y - pos[1]), axis=1, arr=self.sensors
        )
        return np.column_stack((non_zero, np.zeros((2, zeros_len))))

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
        """
        self.sensors = sensors
        self._meas_noise = meas_noise

    def mapping(self, state, time_step=None):
        return np.apply_along_axis(lambda pos: _angle(state[:2], pos), axis=1, arr=self.sensors)

    def meas_noise(self, time_step):
        return self._meas_noise

    def jacobian(self, state, time_step=None):
        zeros_len = state.shape[0] - 2
        x, y = state[0], state[1]
        non_zero = np.apply_along_axis(lambda pos: _angle_jacobian(x - pos[0], y - pos[1]), axis=1, arr=self.sensors)

        return np.column_stack((non_zero, np.zeros((self.sensors.shape[0], zeros_len))))

    def sample(self, states):
        means = self.map_set(states)
        num_samples, D_y = means.shape
        noise = mvn.rvs(mean=np.zeros((D_y,)), cov=self.meas_noise(None), size=num_samples).reshape(means.shape)
        return means + noise


class BearingsVaryingSensors(MeasModel, Differentiable):
    """Special bearings measurement model

    Combines two sep. models
    """

    def __init__(self, default_model, alt_model, alt_model_time_steps):
        self._default_model = default_model
        self._alt_model = alt_model
        self._alt_model_tss = alt_model_time_steps

    def _select_model(self, time_step):
        if time_step in self._alt_model_tss:
            return self._alt_model
        else:
            return self._default_model

    def alt_tss(self):
        return np.array(list(self._alt_model_tss))

    def map_set(self, states, time_step):
        model = self._select_model(time_step)
        return model.map_set(states, time_step)

    def mapping(self, state, time_step):
        model = self._select_model(time_step)
        return model.mapping(state, time_step)

    def meas_noise(self, time_step):
        model = self._select_model(time_step)
        return model._meas_noise

    def jacobian(self, state, time_step=None):
        model = self._select_model(time_step)
        return model.jacobian(state, time_step)


def _euclid_dist(p_1, p_2):
    return np.sqrt(np.sum((p_1 - p_2) ** 2))


def _euclid_dist_jacobian(x, y):
    den = np.sqrt(x ** 2 + y ** 2)
    return np.array([x / den, y / den])


def _angle(p_1, p_2):
    delta_x = p_1[0] - p_2[0]
    delta_y = p_1[1] - p_2[1]
    return np.arctan2(delta_y, delta_x)


def _angle_jacobian(x, y):
    den = x ** 2 + y ** 2
    return np.array([-y / den, x / den])


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

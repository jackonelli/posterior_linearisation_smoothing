"""Extended Kalman filter (EKF)"""
from typing import Union
from src.filter.base import Filter
from src.models.base import Model, MotionModel, MeasModel, Differentiable


class Ekf(Filter):
    """Extended Kalman filter (EKF)"""

    def __init__(self, motion_model: Union[MotionModel, Differentiable], meas_model: Union[MeasModel, Differentiable]):
        """Note that motion and meas models must be of types that inherits from both
        {Motion/Meas}Model as well as from Differentiable
        but there is no clean way of expressing that with python type hints.
        """

        self._motion_model = motion_model
        self._meas_model = meas_model

    def _motion_lin(self, mean, _cov, _time_step):
        F, b = ext_lin(self._motion_model, mean)
        return (F, b, 0)

    def _meas_lin(self, mean, _cov, _time_step):
        H, c = ext_lin(self._meas_model, mean)
        return (H, c, 0)

    def _proc_noise(self, time_step):
        return self._motion_model.proc_noise(time_step)

    def _meas_noise(self, time_step):
        return self._meas_model.meas_noise(time_step)


def ext_lin(model: Union[Model, Differentiable], mean):
    """First order Taylor Linearisation for the extended filters and smoother"""
    jac = model.jacobian(mean)
    offset = model.mapping(mean) - jac @ mean
    return jac, offset


class ExtCache:
    def __init__(self, motion_model, meas_model):
        self._motion_model = motion_model
        self._meas_model = meas_model
        self.motion_lin = None
        self.meas_lin = None

    def update(self, means, _covs):
        self.motion_lin = [(*ext_lin(self._motion_model, mean_k), 0) for mean_k in means]
        self.meas_lin = [(*ext_lin(self._meas_model, mean_k), 0) for mean_k in means]

    def is_initialized(self):
        # TODO: Full
        return self.proc_lin is not None

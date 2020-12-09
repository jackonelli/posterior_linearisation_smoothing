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

    def _motion_lin(self, state, _cov, _time_step):
        F, b = _ekf_lin(self._motion_model, state)
        return (F, b, 0)

    def _meas_lin(self, state, _cov, _time_step):
        H, c = _ekf_lin(self._meas_model, state)
        return (H, c, 0)

    def _proc_noise(self, time_step):
        return self._motion_model.proc_noise(time_step)

    def _meas_noise(self, time_step):
        return self._meas_model.meas_noise(time_step)


def _ekf_lin(model: Union[Model, Differentiable], state):
    jac = model.jacobian(state)
    offset = model.mapping(state) - jac @ state
    return jac, offset

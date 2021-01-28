"""Stochastic coordinated turn motion model"""
import numpy as np
from src.models.base import MotionModel, Differentiable


class CoordTurn(MotionModel, Differentiable):
    """
    state is
        x_k = [
            pos_x,
            pos_y,
            v: speed,
            phi: angle,
            omega: angular_vel,
        ]
    """

    def __init__(self, sampling_period, proc_noise):
        self.sampling_period = sampling_period
        self._proc_noise = proc_noise

    def mapping(self, state, time_step=0):
        v = state[2]
        phi = state[3]
        omega = state[4]
        delta = np.array(
            [
                self.sampling_period * v * np.cos(phi),
                self.sampling_period * v * np.sin(phi),
                0,
                self.sampling_period * omega,
                0,
            ]
        )
        return state + delta

    def proc_noise(self, _k):
        return self._proc_noise

    # TODO: This is not the correct Jacobian
    def jacobian(self, state, time_step=0):
        dt = self.sampling_period
        w = state[4]
        if w == 0:
            coswt = 1
            coswto = 0
            coswtopw = 0
            sinwt = 0
            sinwtpw = dt
            dsinwtpw = 0
            dcoswtopw = -0.5 * dt ** 2
        else:
            coswt = np.cos(w * dt)
            coswto = np.cos(w * dt) - 1
            coswtopw = coswto / w
            sinwt = np.sin(w * dt)
            sinwtpw = sinwt / w
            dsinwtpw = (w * dt * coswt - sinwt) / (w ** 2)
            dcoswtopw = (-w * dt * sinwt - coswto) / (w ** 2)
        jac = np.zeros((5, 5))
        jac[0, 0] = 1
        jac[0, 2] = sinwtpw
        jac[0, 3] = -coswtopw
        jac[0, 4] = dsinwtpw * state[2] - dcoswtopw * state[3]
        jac[1, 1] = 1
        jac[1, 2] = coswtopw
        jac[1, 3] = sinwtpw
        jac[1, 4] = dcoswtopw * state[2] + dsinwtpw * state[3]
        jac[2, 2] = coswt
        jac[2, 3] = sinwt
        jac[2, 4] = -dt * sinwt * state[2] + dt * coswt * state[3]
        jac[3, 2] = -sinwt
        jac[3, 3] = coswt
        jac[3, 4] = -dt * coswt * state[2] - dt * sinwt * state[3]
        jac[4, 4] = 1
        return jac


class LmCoordTurn(MotionModel, Differentiable):
    """
    state is
    TODO: Check definition.
        x_k = [
            pos_x,
            pos_y,
            v: speed,
            phi: angle,
            omega: angular_vel,
        ]
    """

    def __init__(self, sampling_period, proc_noise):
        self._dt = sampling_period
        self._proc_noise = proc_noise

    def mapping(self, state, time_step=0):
        dt = self._dt
        w = state[4]
        if w == 0:
            coswt = np.cos(w * dt)
            coswto = np.cos(w * dt) - 1
            coswtopw = 0
            sinwt = np.sin(w * dt)
            sinwtpw = dt
        else:
            coswt = np.cos(w * dt)
            coswto = np.cos(w * dt) - 1
            coswtopw = coswto / w
            sinwt = np.sin(w * dt)
            sinwtpw = sinwt / w

        F = np.array(
            [
                [1, 0, sinwtpw, -coswtopw, 0],
                [0, 1, coswtopw, sinwtpw, 0],
                [0, 0, coswt, sinwt, 0],
                [0, 0, -sinwt, coswt, 0],
                [0, 0, 0, 0, 1],
            ]
        )
        return F @ state

    def proc_noise(self, _k):
        return self._proc_noise

    def jacobian(self, state):
        dt = self._dt
        w = state[4]
        if w == 0:
            coswt = 1
            coswto = 0
            coswtopw = 0
            sinwt = 0
            sinwtpw = dt
            dsinwtpw = 0
            dcoswtopw = -0.5 * dt ** 2
        else:
            coswt = np.cos(w * dt)
            coswto = np.cos(w * dt) - 1
            coswtopw = coswto / w
            sinwt = np.sin(w * dt)
            sinwtpw = sinwt / w
            dsinwtpw = (w * dt * coswt - sinwt) / (w ** 2)
            dcoswtopw = (-w * dt * sinwt - coswto) / (w ** 2)
        jac = np.zeros((5, 5))
        jac[0, 0] = 1
        jac[0, 2] = sinwtpw
        jac[0, 3] = -coswtopw
        jac[0, 4] = dsinwtpw * state[2] - dcoswtopw * state[3]
        jac[1, 1] = 1
        jac[1, 2] = coswtopw
        jac[1, 3] = sinwtpw
        jac[1, 4] = dcoswtopw * state[2] + dsinwtpw * state[3]
        jac[2, 2] = coswt
        jac[2, 3] = sinwt
        jac[2, 4] = -dt * sinwt * state[2] + dt * coswt * state[3]
        jac[3, 2] = -sinwt
        jac[3, 3] = coswt
        jac[3, 4] = -dt * coswt * state[2] - dt * sinwt * state[3]
        jac[4, 4] = 1
        return jac

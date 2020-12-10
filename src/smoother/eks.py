"""Extended Kalman Smoother (EKS)
Also known as the Extended Rauch-Tung-Striebel Smoother (ERTSS)
"""
from src.smoother.base import Smoother
from src.models.affine import AffineModel
from src.filter.ekf import Ekf, ekf_lin


class Eks(Smoother):
    def __init__(self, motion_model: AffineModel, meas_model: AffineModel):
        super().__init__()
        self._motion_model = motion_model
        self._meas_model = meas_model

    def _motion_lin(self, state, _cov, _time_step):
        F, b = ekf_lin(self._motion_model, state)
        return (F, b, 0)

    def _meas_lin(self, state, _cov, _time_step):
        H, c = ekf_lin(self._meas_model, state)
        return (H, c, 0)

    def _filter_seq(self, measurements, x_0_0, P_0_0):
        return Ekf(self._motion_model, self._meas_model).filter_seq(measurements, x_0_0, P_0_0)

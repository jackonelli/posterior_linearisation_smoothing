"""Rauch-Tung-Striebel Smoother"""
from src.smoother.base import Smoother
from src.models.affine import AffineModel
from src.filter.kalman import KalmanFilter


class RtsSmoother(Smoother):
    def __init__(self, motion_model: AffineModel, meas_model: AffineModel):
        super().__init__()
        self._motion_model = motion_model
        self._meas_model = meas_model

    def _motion_lin(self, _mean, _cov, _time_step):
        return (self._motion_model.linear_map, self._motion_model.offset, 0)

    def _filter_seq(self, measurements, m_1_0, P_1_0):
        return KalmanFilter(self._motion_model, self._meas_model).filter_seq(measurements, m_1_0, P_1_0)

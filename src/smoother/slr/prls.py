"""Sigma point Prior linearisation smoother (PrLS)"""
from src.smoother.base import Smoother
from src.filter.prlf import SigmaPointPrLf
from src.slr.sigma_points import SigmaPointSlr


class SigmaPointPrLs(Smoother):
    """Sigma point PrLs smoother"""

    def __init__(self, motion_model, meas_model, sigma_point_method):
        super().__init__()
        self._motion_model = motion_model
        self._meas_model = meas_model
        self._slr = SigmaPointSlr(sigma_point_method)
        self._sigma_point_method = sigma_point_method

    def _motion_lin(self, mean, cov, time_step):
        return self._slr.linear_params(
            self._mapping_with_time_step(self._motion_model.map_set, time_step=time_step), mean, cov
        )

    def _filter_seq(self, measurements, m_1_0, P_1_0):
        return SigmaPointPrLf(self._motion_model, self._meas_model, self._sigma_point_method).filter_seq(
            measurements, m_1_0, P_1_0
        )

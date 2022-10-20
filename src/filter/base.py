"""Abstract filter class"""
from abc import abstractmethod, ABC
from functools import partial
import logging
import numpy as np

LOGGER = logging.getLogger(__name__)


class Filter(ABC):
    """Abstract filter class

    Assumes motion and meas model on the form:
        x_k = f(x_{k-1}) + q_k, q_k ~ N(0, Q_k)
        y_k = f(x_k}) + r_k, r_k ~ N(0, R_k).

    The filtering is done through linearisation of the motion and meas models:
        f(x) = A x + b + omega, omega ~ N(0, Omega)
        h(x) = H x + c + lambda, lambda ~ N(0, Lambda).

    All the filters in this codebase differs only in their method of linearisation.
    As such, a new filter type is created by specifying the method of linearisation
    in the concrete implementations of this class.
    """

    def filter_seq(self, measurements, m_1_0, P_1_0):
        """Filters a measurement sequence

        Args:
            measurements (K, D_y): Measurement sequence for times 1,..., K
                The measurement seq can be any "iteratable" container of np arrays,
                as long as `len(measurements) = K`.
            m_1_0 (D_x,): Prior mean for time 1
            P_1_0 (D_x, D_x): Prior covariance for time 1

        Returns:
            filter_means (K, D_x): Filtered estimates for times 1,..., K
            filter_covs (K, D_x, D_x): Filter error covariance
            pred_means (K, D_x): Predicted estimates for times 1,..., K
            pred_covs (K, D_x, D_x): Filter error covariance
        """

        K = len(measurements)

        filter_means, filter_covs = self._init_estimates(m_1_0, K)
        pred_means, pred_covs = self._init_estimates(m_1_0, K)

        # first step
        y_1 = measurements[0]
        m_1_1, P_1_1 = self._update(y_1, m_1_0, P_1_0, self._meas_noise(1), self._meas_lin(m_1_0, P_1_0, 1), 1)

        pred_means[0, :] = m_1_0
        pred_covs[0, :, :] = P_1_0
        filter_means[0, :] = m_1_1
        filter_covs[0, :, :] = P_1_1

        # Shift to next time step
        m_kminus1_kminus1 = m_1_1
        P_kminus1_kminus1 = P_1_1
        for k in np.arange(2, K + 1):
            # time step starts at 1, while storing arrays are zero-indexed.
            store_ind = k - 1
            m_k_kminus1, P_k_kminus1 = self._predict(
                m_kminus1_kminus1,
                P_kminus1_kminus1,
                self._proc_noise(k - 1),
                self._motion_lin(m_kminus1_kminus1, P_kminus1_kminus1, k - 1),
            )

            y_k = measurements[store_ind]
            m_k_k, P_k_k = self._update(
                y_k,
                m_k_kminus1,
                P_k_kminus1,
                self._meas_noise(k),
                self._meas_lin(m_k_kminus1, P_k_kminus1, k),
                k,
            )
            pred_means[store_ind, :] = m_k_kminus1
            pred_covs[store_ind, :, :] = P_k_kminus1
            filter_means[store_ind, :] = m_k_k
            filter_covs[store_ind, :, :] = P_k_k
            # Shift to next time step
            m_kminus1_kminus1 = m_k_k
            P_kminus1_kminus1 = P_k_k

        return filter_means, filter_covs, pred_means, pred_covs

    @staticmethod
    def _predict(m_kminus1_kminus1, P_kminus1_kminus1, Q, linearization):
        """KF prediction step

        Args:
            m_kminus1_kminus1: m_{k-1 | k-1}
            P_kminus1_kminus1: P_{k-1 | k-1}
            linearization (tuple): (A, b, Omega) param's for linear (affine) approx

        Returns:
            m_k_kminus1: m_{k | k-1}
            P_k_kminus1: P_{k | k-1}
        """
        A, b, Omega = linearization
        m_k_kminus1 = A @ m_kminus1_kminus1 + b
        P_k_kminus1 = A @ P_kminus1_kminus1 @ A.T + Omega + Q
        P_k_kminus1 = (P_k_kminus1 + P_k_kminus1.T) / 2
        return m_k_kminus1, P_k_kminus1

    def _update(_self, y_k, m_k_kminus1, P_k_kminus1, R, linearization, _time_step):
        """KF update step

        This is a static method in most cases, but e.g. LM-IEKS needs the context of self, and time_step
        when overriding this method.

        Args:
            y_k
            m_k_kminus1: m_{k | k-1}
            P_k_kminus1: P_{k | k-1}
            linearization (tuple): (H, c, Lambda) param's for linear (affine) approx

        Returns:
            m_k_k: m_{k | k}
            P_k_k: P_{k | k}
        """
        if not any(np.isnan(y_k)):
            H, c, Lambda = linearization
            y_mean = H @ m_k_kminus1 + c
            S = H @ P_k_kminus1 @ H.T + R + Lambda
            K = P_k_kminus1 @ H.T @ np.linalg.inv(S)

            m_k_k = m_k_kminus1 + (K @ (y_k - y_mean)).reshape(m_k_kminus1.shape)
            P_k_k = P_k_kminus1 - K @ S @ K.T
            P_k_k = (P_k_k + P_k_k.T) / 2
            return m_k_k, P_k_k
        else:
            return m_k_kminus1, P_k_kminus1

    @abstractmethod
    def _motion_lin(self, state, cov, time_step):
        pass

    @abstractmethod
    def _meas_lin(self, state, cov, time_step):
        pass

    @abstractmethod
    def _meas_noise(self):
        pass

    @abstractmethod
    def _proc_noise(self, time_step):
        pass

    @staticmethod
    def _init_estimates(m_1_0, K):
        D_x = m_1_0.shape[0]
        est_means = np.empty((K, D_x))
        est_covs = np.empty((K, D_x, D_x))
        return est_means, est_covs

    @staticmethod
    def _mapping_with_time_step(mapping, time_step):
        return partial(mapping, time_step=time_step)

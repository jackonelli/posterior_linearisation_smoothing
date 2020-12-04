"""Abstract filter class"""
from abc import abstractmethod, ABC
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

    The method of linearisation is specified in the concrete implementations of this class.
    """

    def filter_seq(self, measurements, x_0_0, P_0_0):
        """Filters a measurement sequence

        Args:
            measurements (K, D_y): Measurement sequence for times 1,..., K
            x_0_0 (D_x,): Prior mean for time 0
            P_0_0 (D_x, D_x): Prior covariance for time 0

        Returns:
            filter_means (K, D_x): Filtered estimates for times 1,..., K
            filter_covs (K, D_x, D_x): Filter error covariance
            pred_means (K, D_x): Predicted estimates for times 1,..., K
            pred_covs (K, D_x, D_x): Filter error covariance
        """

        K = measurements.shape[0]

        filter_means, filter_covs = self._init_estimates(x_0_0, P_0_0, K)
        pred_means, pred_covs = self._init_estimates(x_0_0, P_0_0, K)

        x_kminus1_kminus1 = x_0_0
        P_kminus1_kminus1 = P_0_0
        for k in np.arange(1, K + 1):
            LOGGER.debug("Time step: %s", k)
            # measurement vec is zero-indexed
            # this really gives y_k
            y_k = measurements[k - 1]
            x_k_kminus1, P_k_kminus1 = self._predict(
                x_kminus1_kminus1,
                P_kminus1_kminus1,
                self._process_noise(),
                self._motion_lin(x_kminus1_kminus1, P_kminus1_kminus1, k - 1),
            )

            x_k_k, P_k_k = self._update(
                y_k, x_k_kminus1, P_k_kminus1, self._meas_noise(), self._meas_lin(x_k_kminus1, P_k_kminus1, k)
            )

            pred_means[k, :] = x_k_kminus1
            pred_covs[k, :, :] = P_k_kminus1
            filter_means[k, :] = x_k_k
            filter_covs[k, :, :] = P_k_k
            # Shift to next time step
            x_kminus1_kminus1 = x_k_k
            P_kminus1_kminus1 = P_k_k

        return filter_means, filter_covs, pred_means, pred_covs

    @staticmethod
    def _predict(x_kminus1_kminus1, P_kminus1_kminus1, Q, linearization):
        """KF prediction step

        Args:
            x_kminus1_kminus1: x_{k-1 | k-1}
            P_kminus1_kminus1: P_{k-1 | k-1}
            linearization (tuple): (A, b, Omega) param's for linear (affine) approx

        Returns:
            x_k_kminus1: x_{k | k-1}
            P_k_kminus1: P_{k | k-1}
        """
        A, b, Omega = linearization
        x_k_kminus1 = A @ x_kminus1_kminus1 + b
        P_k_kminus1 = A @ P_kminus1_kminus1 @ A.T + Omega + Q
        P_k_kminus1 = (P_k_kminus1 + P_k_kminus1.T) / 2
        return x_k_kminus1, P_k_kminus1

    @staticmethod
    def _update(y_k, x_k_kminus1, P_k_kminus1, R, linearization):
        """KF update step
        Args:
            y_k
            x_k_kminus1: x_{k | k-1}
            P_k_kminus1: P_{k | k-1}
            linearization (tuple): (H, c, Lambda) param's for linear (affine) approx

        Returns:
            x_k_k: x_{k | k}
            P_k_k: P_{k | k}
        """
        H, c, Lambda = linearization
        y_mean = H @ x_k_kminus1 + c
        S = H @ P_k_kminus1 @ H.T + R + Lambda
        K = P_k_kminus1 @ H.T @ np.linalg.inv(S)

        x_k_k = x_k_kminus1 + (K @ (y_k - y_mean)).reshape(x_k_kminus1.shape)
        P_k_k = P_k_kminus1 - K @ S @ K.T

        return x_k_k, P_k_k

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
    def _process_noise(self):
        pass

    @staticmethod
    def _init_estimates(x_0_0, P_0_0, K):
        D_x = x_0_0.shape[0]
        est_means = np.empty((K + 1, D_x))
        est_covs = np.empty((K + 1, D_x, D_x))
        est_means[0, :] = x_0_0
        est_covs[0, :, :] = P_0_0
        return est_means, est_covs

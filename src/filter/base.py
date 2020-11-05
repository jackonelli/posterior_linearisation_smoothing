"""Abstract filter class"""
from abc import abstractmethod, ABC
import logging
import numpy as np

LOGGER = logging.getLogger(__name__)


class Filter(ABC):
    """Abstract filter class
    # TODO: Need a name for this. Gaussian filter?
    """

    def filter_seq(
        self,
        measurements,
        x_0_0,
        P_0_0,
    ):
        """Kalman filter with general linearization
        Filters a measurement sequence using a linear Kalman filter.

        Args:
            measurements (K, D_y): Measurement sequence for times 1,..., K
            x_0_0 (D_x,): Prior mean for time 0
            P_0_0 (D_x, D_x): Prior covariance for time 0
            motion_lin: Must inherit from Linearizer
            meas_lin: Must inherit from Linearizer

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
                x_kminus1_kminus1, P_kminus1_kminus1, self._motion_lin(x_kminus1_kminus1, P_kminus1_kminus1)
            )

            x_k_k, P_k_k = self._update(y_k, x_k_kminus1, P_k_kminus1, self._meas_lin(x_k_kminus1, P_k_kminus1))

            pred_means[k, :] = x_k_kminus1
            pred_covs[k, :, :] = P_k_kminus1
            filter_means[k, :] = x_k_k
            filter_covs[k, :, :] = P_k_k
            # Shift to next time step
            x_kminus1_kminus1 = x_k_k
            P_kminus1_kminus1 = P_k_k

        return filter_means, filter_covs, pred_means, pred_covs

    @staticmethod
    def _predict(x_kminus1_kminus1, P_kminus1_kminus1, linearization):
        """KF prediction step

        Args:
            x_kminus1_kminus1: x_{k-1 | k-1}
            P_kminus1_kminus1: P_{k-1 | k-1}
            linearization (tuple): (A, b, Q) param's for linear (affine) approx

        Returns:
            x_k_kminus1: x_{k | k-1}
            P_k_kminus1: P_{k | k-1}
        """
        A, b, Omega = linearization
        x_k_kminus1 = A @ x_kminus1_kminus1 + b
        P_k_kminus1 = A @ P_kminus1_kminus1 @ A.T + Omega
        P_k_kminus1 = (P_k_kminus1 + P_k_kminus1.T) / 2
        return x_k_kminus1, P_k_kminus1

    @staticmethod
    def _update(y_k, x_k_kminus1, P_k_kminus1, linearization):
        """KF update step
        Args:
            y_k
            x_k_kminus1: x_{k | k-1}
            P_k_kminus1: P_{k | k-1}
            linearization (tuple): (H, c, R) param's for linear (affine) approx

        Returns:
            x_k_k: x_{k | k}
            P_k_k: P_{k | k}
        """
        H, c, R = linearization
        y_mean = H @ x_k_kminus1 + c
        # S.shape = (D_y, D_y)
        S = H @ P_k_kminus1 @ H.T + R
        # K.shape = (D_x, D_y)
        K = P_k_kminus1 @ H.T @ np.linalg.inv(S)

        x_k_k = x_k_kminus1 + (K @ (y_k - y_mean)).reshape(x_k_kminus1.shape)
        P_k_k = P_k_kminus1 - K @ S @ K.T

        return x_k_k, P_k_k

    @abstractmethod
    def _meas_lin(self, state, cov):
        pass

    @abstractmethod
    def _motion_lin(self, state, cov):
        pass

    @staticmethod
    def _init_estimates(x_0_0, P_0_0, K):
        D_x = x_0_0.shape[0]
        est_means = np.empty((K + 1, D_x))
        est_covs = np.empty((K + 1, D_x, D_x))
        est_means[0, :] = x_0_0
        est_covs[0, :, :] = P_0_0
        return est_means, est_covs

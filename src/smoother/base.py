"""Abstract smoother class"""
from abc import abstractmethod, ABC
import logging
import numpy as np
from src.filter.base import Filter


class Smoother(ABC):
    """Abstract smoother class

    Assumes motion and meas model on the form:
        x_k = f(x_{k-1}) + q_k, q_k ~ N(0, Q_k)
        y_k = f(x_k}) + r_k, r_k ~ N(0, R_k).

    All the smoother in this codebase differs only in their method of linearisation.
    As such, a new filter type is created by specifying the method of linearisation
    in the concrete implementations of this class.
    """

    def __init__(self):
        self._log = logging.getLogger(self.__class__.__name__)

    def filter_and_smooth(self, measurements, m_1_0, P_1_0, cost_fn=None):
        """Filters and smooths a measurement sequence.

        Args:
            measurements (K, D_y): Measurement sequence for times 1,..., K
            m_0_0 (D_x,): Prior mean for time 1
            P_0_0 (D_x, D_x): Prior covariance for time 1
            cost_fn: optional fn, mapping estimated traj to cost: R^(K x D_x) --> R
                useful to track progress for iterated smoother and needs to be included here
                to get a uniform api.

        Returns:
            filter_means (K, D_x): Filtered mean estimates for times 1,..., K
            filter_covs (K, D_x, D_x): Filtered covariance estimates for times 1,..., K
            smooth_means (K, D_x): Smoothed mean estimates for times 1,..., K
            smooth_covs (K, D_x, D_x): Smoothed covariance estimates for times 1,..., K
        """

        filter_means, filter_covs, pred_means, pred_covs = self._filter_seq(measurements, m_1_0, P_1_0)
        smooth_means, smooth_covs = self.smooth_seq_pre_comp_filter(filter_means, filter_covs, pred_means, pred_covs)
        cost = None
        if cost_fn is not None:
            cost = cost_fn(smooth_means)
        return filter_means, filter_covs, smooth_means, smooth_covs, cost

    def smooth_seq_pre_comp_filter(self, filter_means, filter_covs, pred_means, pred_covs):
        """Smooths the outputs from a filter.

        Args:
            filter_means (K, D_x): Filtered estimates for times 1,..., K
            filter_covs (K, D_x, D_x): Filtered covariance estimates for times 1,..., K
            pred_means (K, D_x): Predicted estimates for times 1,..., K
            pred_covs (K, D_x, D_x): Predicted covariance estimates for times 1,..., K

        Returns:
            smooth_means (K, D_x): Smoothed estimates for times 1,..., K
            smooth_covs (K, D_x, D_x): Smoothed covariance estimates for times 1,..., K
        """

        K = filter_means.shape[0]
        smooth_means, smooth_covs = self._init_smooth_estimates(filter_means[-1, :], filter_covs[-1, :, :], K)
        for k in np.flip(np.arange(1, K)):
            m_kminus1_kminus1 = filter_means[k - 1, :]
            P_kminus1_kminus1 = filter_covs[k - 1, :, :]
            m_k_K, P_k_K = smooth_means[k, :], smooth_covs[k, :, :]
            m_k_kminus1, P_k_kminus1 = pred_means[k, :], pred_covs[k, :, :]
            m_kminus1_K, P_kminus1_K = self._rts_update(
                m_k_K,
                P_k_K,
                m_kminus1_kminus1,
                P_kminus1_kminus1,
                m_k_kminus1,
                P_k_kminus1,
                self._motion_lin(m_kminus1_kminus1, P_kminus1_kminus1, k - 1),
            )
            smooth_means[k - 1, :] = m_kminus1_K
            smooth_covs[k - 1, :, :] = P_kminus1_K
        return smooth_means, smooth_covs

    @abstractmethod
    def _filter_seq(self, measurements, m_1_0, P_1_0):
        """Filter sequence

        Technically smoothers do not require the ability to filter.
        Given a motion model and filtered and predicted estimates,
        the smooth estimates can be calculated without an explicit filter and meas model.
        However, a concrete implementation of a smoother, e.g. the RTS smoother,
        commonly smooths estimates coming from a KF, not some other filter.

        The API still allows for smoothing of any sequence by using the `smooth_seq_pre_comp_filter` method
        """
        pass

    def _rts_update(self, m_k_K, P_k_K, m_kminus1_kminus1, P_kminus1_kminus1, m_k_kminus1, P_k_kminus1, linear_params):
        """RTS update step
        Args:
            m_k_K: m_{k|K}
            P_k_K: P_{k|K}
            m_kminus1_kminus1: m_{k-1 | k-1}
            P_kminus1_kminus1: P_{k-1 | k-1}
            m_k_kminus1: m_{k | k-1}
            P_k_kminus1: P_{k | k-1}
            linearization (tuple): (A, b, Q) param's for linear (affine) approx

        Returns:
            m_kminus1_K: m_{k-1 | K}
            P_kminus1_K: P_{k-1 | K}
        """
        A, _, Q = linear_params

        G_k = P_kminus1_kminus1 @ A.T @ np.linalg.inv(P_k_kminus1)
        m_kminus1_K = m_kminus1_kminus1 + G_k @ (m_k_K - m_k_kminus1)
        P_kminus1_K = P_kminus1_kminus1 + G_k @ (P_k_K - P_k_kminus1) @ G_k.T
        return m_kminus1_K, P_kminus1_K

    @staticmethod
    def _init_smooth_estimates(m_K_K, P_K_K, K):
        D_x = m_K_K.shape[0]
        smooth_means = np.empty((K, D_x))
        smooth_covs = np.empty((K, D_x, D_x))
        smooth_means[-1, :] = m_K_K
        smooth_covs[-1, :, :] = P_K_K
        return smooth_means, smooth_covs

    @abstractmethod
    def _motion_lin(state, cov, time_step):
        """Linearise motion model

        Time step k gives required context for some linearisations (Posterior SLR).
        """
        pass


class IteratedSmoother(Smoother):
    """Abstract iterated smoother class

    The purpose is to provide a default impl. for the high level method
    `smooth_and_filter_iter`
    """

    def filter_and_smooth(self, measurements, m_1_0, P_1_0, cost_fn):
        """Overrides (extends) the base class default implementation"""

        mf, Pf, current_ms, current_Ps, first_cost = self._first_iter(measurements, m_1_0, P_1_0, cost_fn)
        iter_cost = np.array([first_cost])
        if self.num_iter > 1:
            mf, Pf, ms, Ps, tmp_cost = self.filter_and_smooth_with_init_traj(
                measurements, m_1_0, P_1_0, (current_ms, current_Ps), 2, cost_fn
            )
            return mf, Pf, ms, Ps, np.concatenate((iter_cost, tmp_cost))
        else:
            return mf, Pf, current_ms, current_Ps, iter_cost

    def filter_and_smooth_with_init_traj(self, measurements, m_1_0, P_1_0, init_traj, start_iter, cost_fn):
        """Filter and smoothing given an initial trajectory

        Override if more complex iteration behaviour decided (e.g. reject iter based on cost fn increase)
        TODO: Can this be made to make overrides unnec? Adding rejection predicate perhaps.
        """
        current_ms, current_Ps = init_traj
        self._update_estimates(current_ms, current_Ps)
        cost_iter = [cost_fn(current_ms)]
        for iter_ in range(start_iter, self.num_iter + 1):
            self._log.info(f"Iter: {iter_}")
            mf, Pf, current_ms, current_Ps, cost = super().filter_and_smooth(measurements, m_1_0, P_1_0, cost_fn)
            self._update_estimates(current_ms, current_Ps)
            cost_iter.append(cost)
        return mf, Pf, current_ms, current_Ps, np.array(cost_iter)

    @abstractmethod
    def _update_estimates(means, covs):
        """The 'previous estimates' which are used in the current iteration are stored in the smoother instance.
        They should only be modified through this method.
        """
        pass

    @abstractmethod
    def _first_iter(measurements, m_1_0, P_1_0, cost_fn):
        """First, special, iter to initialise the 'previous estimates'"""
        pass

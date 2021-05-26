"""Abstract smoother class"""
from abc import abstractmethod, ABC
from functools import partial
import logging
import numpy as np


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
                # TODO: k should be 'k-1' here? Or not maybe
                self._motion_lin(m_kminus1_kminus1, P_kminus1_kminus1, k),
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

    @staticmethod
    def _mapping_with_time_step(mapping, time_step):
        return partial(mapping, time_step=time_step)


class IteratedSmoother(Smoother):
    """Abstract iterated smoother class

    The purpose is to provide a default impl. for the high level method
    `smooth_and_filter_iter`
    """

    def __init__(self):
        super().__init__()
        self._current_means = None
        self._current_covs = None
        self._precomp_linearisations = []
        self._store_means = []
        self._store_covs = []

    def filter_and_smooth(self, measurements, m_1_0, P_1_0, cost_fn):
        """Overrides (extends) the base class default implementation"""

        self._log.debug("Iter: 1")
        mf, Pf, current_ms, current_Ps, initial_cost = self._first_iter(measurements, m_1_0, P_1_0, cost_fn)
        self._log.debug(f"Initial cost: {initial_cost}")
        if self.num_iter > 1:
            mf, Pf, ms, Ps, tmp_cost = self.filter_and_smooth_with_init_traj(
                measurements, m_1_0, P_1_0, (current_ms, current_Ps), 2, cost_fn
            )
            return mf, Pf, ms, Ps, np.insert(tmp_cost, 0, initial_cost)
        else:
            iter_cost = np.array([initial_cost])
            if not self._is_initialised():
                self._update_estimates(current_ms, current_Ps)
            return mf, Pf, current_ms, current_Ps, iter_cost

    def filter_and_smooth_with_init_traj(self, measurements, m_1_0, P_1_0, init_traj, start_iter, cost_fn_prototype):
        """Filter and smoothing given an initial trajectory

        Override if more complex iteration behaviour decided (e.g. reject iter based on cost fn increase)
        TODO: Can this be made to make overrides unnec? Adding rejection predicate perhaps.
        """
        current_ms, current_Ps = init_traj
        # If self.num_iter is too low to enter the iter loop
        mf, Pf = init_traj
        if not self._is_initialised():
            self._update_estimates(current_ms, current_Ps)
        cost_iter = []
        cost_fn = self._specialise_cost_fn(cost_fn_prototype, self._cost_fn_params())
        for iter_ in range(start_iter, self.num_iter + 1):
            self._log.debug(f"Iter: {iter_}")
            mf, Pf, current_ms, current_Ps, _ = super().filter_and_smooth(measurements, m_1_0, P_1_0, cost_fn)
            self._update_estimates(current_ms, current_Ps)
            cost_fn = self._specialise_cost_fn(cost_fn_prototype, self._cost_fn_params())
            cost = cost_fn(current_ms)
            self._log.debug(f"Cost: {cost}")
            cost_iter.append(cost)
        return mf, Pf, current_ms, current_Ps, np.array(cost_iter)

    @abstractmethod
    def _first_iter(measurements, m_1_0, P_1_0, cost_fn):
        """First, special, iter to initialise the 'previous estimates'"""
        pass

    def _specialise_cost_fn(self, cost_fn_prototype, params):
        """Required for methods which update the cost fn, e.g. Reg-IPLS."""
        return cost_fn_prototype

    def _cost_fn_params(self):
        """Extra parameters used for specialising the cost fn"""
        return None

    def _update_estimates(self, means, covs):
        """The 'previous estimates' which are used in the current iteration are stored in the smoother instance.
        They should only be modified through this method.
        """
        self._current_means = means.copy()
        self._store_means.append(means)
        if covs is not None:
            self._current_covs = covs.copy()
        self._store_covs.append(covs)

    def stored_estimates(self):
        for means, covs in zip(self._store_means, self._store_covs):
            yield means, covs

    @abstractmethod
    def _is_initialised(self):
        pass

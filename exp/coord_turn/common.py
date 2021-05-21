from enum import Enum
import numpy as np
from src.analytics import rmse, nees


class MeasType(Enum):
    Range = "range"
    Bearings = "bearings"


def mc_stats(data):
    num_mc_samples = data.shape[0]
    return np.mean(data, 0), np.std(data, 0) / np.sqrt(num_mc_samples)


def run_smoothing(smoother, states, measurements, prior_mean, prior_cov, cost_fn, init_traj=None):
    """Common function that runs a smoother and collects metrics

    Some iterative smoothers may return early if they exceed the limit on the number of loss-improving trials.
    In those cases, the metrics are extended with the last element to a list of length `smoother.num_iter`
    """
    if init_traj is not None:
        _, _, ms, Ps, iter_cost = smoother.filter_and_smooth_with_init_traj(
            measurements, prior_mean, prior_cov, init_traj, 1, cost_fn
        )
        stored_est = smoother.stored_estimates()
        next(stored_est)
        stored_est = list(stored_est)
    else:
        _, _, ms, Ps, iter_cost = smoother.filter_and_smooth(measurements, prior_mean, prior_cov, cost_fn)
        stored_est = list(smoother.stored_estimates())
    rmses = calc_iter_metrics(
        lambda means, covs, states: rmse(means[:, :-1], states), stored_est, states, smoother.num_iter
    )
    # assert np.allclose(ms_st, ms)
    neeses = calc_iter_metrics(
        lambda means, covs, states: np.mean(nees(states, means[:, :-1], covs[:, :-1, :-1])),
        stored_est,
        states,
        smoother.num_iter,
    )
    return ms, Ps, iter_cost, rmses, neeses


def calc_iter_metrics(metric_fn, estimates, states, num_iter):
    metrics = np.array([metric_fn(means, covs, states) for means, covs in estimates])
    if len(metrics) < num_iter:
        metrics = np.concatenate(
            (
                metrics,
                metrics[-1]
                * np.ones(
                    num_iter - len(metrics),
                ),
            )
        )
    return metrics
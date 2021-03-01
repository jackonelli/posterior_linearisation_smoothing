"""Data for the experiment in the paper "Iteraterd posterior linearization smoother"
"""
from pathlib import Path
from typing import Tuple
import numpy as np
from src.models.nonstationary_growth import NonStationaryGrowth
from src.models.base import MeasModel


def simulate_data(
    num_steps: int,
    prior_mean: float,
    prior_cov: float,
    motion_model: NonStationaryGrowth,
    meas_model: MeasModel,
) -> Tuple[np.ndarray, np.ndarray]:
    chol_ini = np.sqrt(prior_cov)
    chol_Q = np.sqrt(motion_model.proc_noise(None))

    states = np.empty((num_steps, 1))
    xk = prior_mean + chol_ini * np.random.randn()
    for k in range(1, num_steps):
        xk_pred = motion_model.mapping(xk, k) + chol_Q * np.random.randn()
        states[k] = xk_pred
        xk = xk_pred

    chol_R = np.sqrt(meas_model.meas_noise(None))
    meas_noise = chol_R * np.random.randn(num_steps, 1)
    meas = meas_model.map_set(states) + meas_noise
    return states, meas


def simulate_multiple_trajs(
    num_steps: int,
    num_mc_runs: int,
    num_mc_per_traj: int,
    prior_mean: float,
    prior_cov: float,
    model: NonStationaryGrowth,
) -> Tuple[np.ndarray, np.ndarray]:
    num_trajs = num_mc_runs // num_mc_per_traj

    chol_ini = np.sqrt(prior_cov)
    chol_Q = np.sqrt(model.proc_noise(None))

    X_multi_series = np.empty((num_trajs, num_steps))

    for i in range(num_trajs):
        X_multi_i = np.empty((num_steps,))
        xk = prior_mean + chol_ini * np.random.randn()
        X_multi_i[0] = xk
        for k in range(1, num_steps):
            (xk_pred, _) = model.mapping((xk, k)) + chol_Q * np.random.randn()
            X_multi_i[k] = xk_pred
            xk = xk_pred
        X_multi_series[i, :] = X_multi_i

    noise_z = np.random.randn(num_steps * num_mc_runs)
    return X_multi_series, noise_z


def get_specific_states_from_file(data_root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states_file = data_root / "states.csv"
    if states_file.exists():
        states = np.genfromtxt(states_file, dtype=float, delimiter=";", comments="#")
    else:
        raise FileNotFoundError(f"No states data file at '{states_file}'")

    noise_file = data_root / "noise.csv"
    if noise_file.exists():
        noise = np.genfromtxt(noise_file, dtype=float, delimiter=";", comments="#")
    else:
        raise FileNotFoundError(f"No noise data file at '{noise_file}'")

    xs_file = data_root / "xs_1.csv"
    if xs_file.exists():
        xs = np.genfromtxt(xs_file, dtype=float, delimiter=";", comments="#")
    else:
        raise FileNotFoundError(f"No means data file at '{xs_file}'")

    Ps_file = data_root / "Ps_1.csv"
    if Ps_file.exists():
        Ps = np.genfromtxt(Ps_file, dtype=float, delimiter=";", comments="#")
    else:
        raise FileNotFoundError(f"No covs data file at '{Ps_file}'")

    return states, noise.T, xs, Ps


def gen_measurements(states: np.ndarray, standard_normal_noise: np.ndarray, meas_model: MeasModel):
    num_time_steps = states.shape[0]
    standard_normal_noise = standard_normal_noise.reshape((num_time_steps, 1))
    time_step = None
    std = meas_model.meas_noise(time_step)
    noise = std * standard_normal_noise
    return meas_model.map_set(states, time_step) + noise


if __name__ == "__main__":
    K = 50
    num_mc_runs = 1000  # 1000 in original exp
    num_mc_per_traj = 50
    np.random.seed(0)
    model = NonStationaryGrowth(alpha=0.9, beta=10, gamma=8, delta=1.2, proc_noise=1)
    trajs, noise = get_specific_states_from_file(Path.cwd() / "data/ipls_paper")
    print(gen_measurements(trajs[:, 0], noise[0, :]))

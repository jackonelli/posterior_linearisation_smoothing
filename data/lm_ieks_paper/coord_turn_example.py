""" LM-IEKS coordinated turn simulation

Data simulation used in the paper
Levenberg-marquardt and line-search extended kalman smoother
The particular realisation used in the paper is data/lm_ieks_coord_turn_states.csv
"""

from enum import Enum
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from matplotlib import pyplot as plt
from src.models.range_bearing import RangeBearing, MultiSensorRange


class Type(Enum):
    Extended = "ext"
    GN = "gn"
    LM = "lm"


def gen_data(sens_pos_1, sens_pos_2, std, dt, x_0, time_steps, seed=None) -> (np.ndarray, np.ndarray):
    """Create a curved trajectory and angle measurements from two sensors (TODO angles)
    D_x = 4, D_y = 2

    LM-IEKS paper parameters:
        sens_pos_1 = np.array([-1.5, 0.5])
        sens_pos_2 = np.array([1, 1])
        std = 0.5
        dt = 0.01
        x_0 = np.array([0.1, 0.2, 1, 0])
        time_steps = 500
        seed = 4 # NB. Matlab seed

    Args:
        sens_pos_1 (np.ndarray): Position of sensor 1 (2,)
        sens_pos_2 (np.ndarray): Position of sensor 2 (2,)
        std (float): Standard deviation of measurements
        dt (float): Sampling period
        x_0 (np.ndarray): Initial state (D_x,)
        time_steps (int): Number of timesteps to simulate
        seed(int): random seed

    Returns:
        data (np.ndarray): (time_steps, D_x)
    """
    if seed is not None:
        np.random.seed(seed)

    D_x = x_0.shape[0]
    D_y = sens_pos_1.shape[0]
    a = 1 + dt * 10 * np.cumsum(np.random.randn(1, time_steps))
    meas_model_1 = RangeBearing(sens_pos_1, std)
    meas_model_2 = RangeBearing(sens_pos_2, std)
    # meas_model = MultiSensorRange(np.row_stack((sens_pos_1, sens_pos_2)), std)

    x = x_0
    t = 0
    states = np.empty((time_steps, D_x))
    range_meas = np.empty((time_steps, D_y))
    T = []
    for k in range(time_steps):
        F = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, a[k]], [0, 0, -a[k], 0]])
        x = expm(F * dt) @ x
        y1 = meas_model_1.mapping(x)[0] + std * np.random.rand()
        y2 = meas_model_2.mapping(x)[0] + std * np.random.rand()
        t += dt
        states[k, :] = x
        T.append(t)
        range_meas[k, :] = np.array([y1, y2])
    # TODO: add sample method on base class.
    # measurements = meas_model.map_set(states)

    return states, range_meas


def get_specific_states_from_file(data_root: Path, type_: Type) -> np.ndarray:
    states_file = data_root / "states.csv"
    if states_file.exists():
        states = np.genfromtxt(states_file, dtype=float, delimiter=";", comments="#")
    else:
        raise FileNotFoundError(f"No states data file at '{states_file}'")

    meas_file = data_root / "meas.csv"
    if meas_file.exists():
        measurements = np.genfromtxt(meas_file, dtype=float, delimiter=";", comments="#")
    else:
        raise FileNotFoundError(f"No meas data file at '{meas_file}'")

    xf_file = data_root / type_.value / "xf.csv"
    if xf_file.exists():
        xf = np.genfromtxt(xf_file, dtype=float, delimiter=";", comments="#")
    else:
        raise FileNotFoundError(f"No xf data file at '{xf_file}'")

    xs_file = data_root / type_.value / "xs.csv"
    if xs_file.exists():
        xs = np.genfromtxt(xs_file, dtype=float, delimiter=";", comments="#")
    else:
        raise FileNotFoundError(f"No xs data file at '{xs_file}'")

    return states, measurements, xf, xs


if __name__ == "__main__":
    data = gen_data(
        sens_pos_1=np.array([-1.5, 0.5]),
        sens_pos_2=np.array([1, 1]),  # Position of sensor 2
        std=0.5,  # Standard deviation of measurements
        dt=0.01,  # Sampling period
        x_0=np.array([0.1, 0.2, 1, 0]),  # Initial state
        time_steps=500,
        seed=4,
    )

    data = get_specific_states_from_file(Path.cwd())

    plt.plot(data[:, 0], data[:, 1])
    plt.show()

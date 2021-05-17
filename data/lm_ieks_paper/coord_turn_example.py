""" LM-IEKS coordinated turn simulation

Data simulation used in the paper
Levenberg-marquardt and line-search extended kalman smoother
The particular realisation used in the paper is data/lm_ieks_coord_turn_states.csv
"""

from enum import Enum
from typing import Optional, Tuple, Union
import logging
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from matplotlib import pyplot as plt
from src.models.range_bearing import RangeBearing, MultiSensorRange, MultiSensorBearings
from src.models.coord_turn import CoordTurn

LOGGER = logging.getLogger(__name__)


class Type(Enum):
    Extended = "ext"
    GN = "gn"
    LM = "lm"
    LineSearch = "line_search"


def simulate_data(
    motion_model: CoordTurn,
    meas_model: Union[MultiSensorRange, MultiSensorBearings],
    x_0,
    time_steps,
    seed=None,
) -> Tuple[np.ndarray, np.ndarray]:
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
        states (np.ndarray): (time_steps, D_x)
        meas (np.ndarray): (time_steps, D_y)
    """
    if seed is not None:
        np.random.seed(seed)

    D_x = x_0.shape[0]
    dt = motion_model._dt
    a = 1 + dt * 10 * np.cumsum(np.random.randn(1, time_steps))

    x = x_0
    t = 0
    states = np.empty((time_steps, D_x))
    for k in range(time_steps):
        # TODO: use motion model
        F = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, a[k]], [0, 0, -a[k], 0]])
        x = expm(F * dt) @ x
        t += dt
        states[k, :] = x
    new_meas = meas_model.sample(states)
    return states, new_meas


def get_specific_states_from_file(
    data_root: Path, type_: Type, num_iter: Optional[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    xf_file = data_root / type_.value / xf_name(num_iter)
    if xf_file.exists():
        xf = np.genfromtxt(xf_file, dtype=float, delimiter=";", comments="#")
    else:
        LOGGER.warn(f"No xf data file at '{xf_file}'")
        xf = None

    xs_file = data_root / type_.value / xs_name(num_iter)
    if xs_file.exists():
        xs = np.genfromtxt(xs_file, dtype=float, delimiter=";", comments="#")
    else:
        LOGGER.warn(f"No xs data file at '{xs_file}'")
        xs = None

    return states, measurements, xf, xs


def xf_name(num_iter: Optional[int]) -> str:
    if num_iter is not None:
        return f"xf_{num_iter}.csv"
    else:
        return f"xf.csv"


def xs_name(num_iter: Optional[int]) -> str:
    if num_iter is not None:
        return f"xs_{num_iter}.csv"
    else:
        return f"xs.csv"

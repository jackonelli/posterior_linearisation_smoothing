from pathlib import Path
from functools import partial
import numpy as np
from scipy.stats import multivariate_normal as mvn
from src.models.coord_turn import CoordTurn
from src.models.range_bearing import RangeBearing
from src.utils import tikz_2d_tab_to_file
from src.models.range_bearing import to_cartesian_coords


def get_states_and_meas(meas_model, R, range_, tunnel_segment):
    true_states = np.loadtxt("data/tunnel_traj.csv")
    start, stop = range_
    meas = gen_meas(true_states, tunnel_segment, meas_model, R)
    # assert tunnel_segment in range_
    return true_states[start:stop], meas[start:stop]


def gen_meas(states, tunnel_segment, meas_model, R):
    """Generate non-linear measurements

    Args:
        states np.array((K, D_x))
        meas_model
        R np.array((D_y, D_y))
    """

    meas_mean = meas_model.map_set(states)
    num_states, meas_dim = meas_mean.shape
    noise = mvn.rvs(mean=np.zeros((meas_dim,)), cov=R, size=num_states)
    meas = meas_mean + noise
    # No meas in tunnel.
    tunnel_start, tunnel_stop = tunnel_segment
    if tunnel_start is not None and tunnel_stop is not None:
        meas[tunnel_start:tunnel_stop, :] = None
    return meas


if __name__ == "__main__":
    # Meas model
    pos = np.array([100, -100])
    # sigma_r = 2
    # sigma_phi = 0.5 * np.pi / 180
    sigma_r = 4
    sigma_phi = 1 * np.pi / 180

    R = np.diag([sigma_r ** 2, sigma_phi ** 2])
    meas_model = RangeBearing(pos, R)

    # Generate data
    range_ = (0, None)
    tunnel_segment = [145, 165]
    states, measurements = get_states_and_meas(meas_model, R, range_, tunnel_segment)
    cartes_meas = np.apply_along_axis(partial(to_cartesian_coords, pos=pos), 1, measurements)
    tikz_2d_tab_to_file([("states", states), ("meas", cartes_meas)], Path("../paper/fig/tunnel_sim/"))

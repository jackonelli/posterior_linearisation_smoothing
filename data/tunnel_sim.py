import numpy as np
from scipy.stats import multivariate_normal as mvn
from src.models.coord_turn import CoordTurn
import matplotlib.pyplot as plt
from src.models.range_bearing import MultiSensorRange


def gen_states():
    K = 50
    t = np.arange(K)
    x = np.sin(np.pi * t / K)
    y = t / K
    return np.column_stack((x, y))


def get_tricky_data(meas_model, R, range_):
    true_states = np.loadtxt("data/ct_data.csv")
    start, stop = range_
    return true_states, gen_non_lin_meas(true_states[start:stop, :], meas_model, R)


def gen_dummy_data(num_samples, sampling_period, meas_model, R):
    omega = np.zeros((num_samples + 1))
    omega[200:401] = -np.pi / 201 / sampling_period
    # Initial state
    initial_state = np.array([0, 0, 20, 0, omega[0]])
    # Allocate memory
    true_states = np.zeros((num_samples + 1, initial_state.shape[0]))
    true_states[0, :] = initial_state
    coord_turn = CoordTurn(sampling_period, None)
    # Create true track
    for k in range(1, num_samples + 1):
        new_state = coord_turn.mean(true_states[k - 1, :])
        true_states[k, :] = new_state
        true_states[k, 4] = omega[k]

    return true_states, gen_non_lin_meas(true_states, meas_model, R)


def gen_non_lin_meas(states, meas_model):
    """Generate non-linear measurements

    Args:
        states np.array((K, D_x))
        meas_model
    """

    meas_mean = meas_model.map_set(states)
    num_states, meas_dim = meas_mean.shape
    noise = mvn.rvs(mean=np.zeros((meas_dim,)), cov=meas_model.meas_noise(None), size=num_states)
    return meas_mean + noise


def main():
    states = gen_states()
    sens_pos_1 = np.array([-2, 1])
    sens_pos_2 = np.array([2, 0.5])
    # sensors = np.row_stack((sens_pos_1, sens_pos_2))
    # std = 0.5
    # R = std ** 2 * np.eye(2)
    # meas_model = MultiSensorRange(sensors, R)
    # meas = gen_non_lin_meas(states, meas_model)
    plt.plot(states[:, 0], states[:, 1])
    plt.plot(sens_pos_1[0], sens_pos_1[1], "rx")
    plt.plot(sens_pos_2[0], sens_pos_2[1], "rx")
    plt.show()


if __name__ == "__main__":
    main()

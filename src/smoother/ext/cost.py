import numpy as np


def cost(means, measurements, m_1_0, P_1_0, motion_model, meas_model):
    prior_diff = means[0, :] - m_1_0
    _cost = prior_diff.T @ np.linalg.inv(P_1_0) @ prior_diff

    proc_diff = means[1:, :] - motion_model.map_set(means[:-1, :])
    meas_diff = measurements - meas_model.map_set(means)
    for k in range(0, means.shape[0] - 1):
        _cost += proc_diff[k, :].T @ np.linalg.inv(motion_model.proc_noise(k)) @ proc_diff[k, :]
        # measurements are zero indexed, i.e. k-1 --> y_k
        _cost += meas_diff[k, :].T @ np.linalg.inv(meas_model.meas_noise(k)) @ meas_diff[k, :]
    _cost += meas_diff[-1, :].T @ np.linalg.inv(meas_model.meas_noise(k)) @ meas_diff[-1, :]

    return _cost

import numpy as np


def lm_cost(means, measurements, m_1_0, P_1_0, motion_model, meas_model):
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


def ss_cost(means, measurements, m_1_0, P_1_0, Q, R, f_fun, h_fun):
    J = (means[0, :] - m_1_0) @ np.linalg.inv(P_1_0) @ (means[0, :] - m_1_0)
    for k in range(0, means.shape[0]):
        x_k = means[k, :]
        z_k = measurements[k, :]
        if k > 0:
            x_k_min_1 = means[k - 1, :]
            J += (x_k - f_fun(x_k_min_1)).T @ np.linalg.inv(Q) @ (x_k - f_fun(x_k_min_1))
        J += (z_k - h_fun(x_k)).T @ np.linalg.inv(R) @ (z_k - h_fun(x_k))
    return J

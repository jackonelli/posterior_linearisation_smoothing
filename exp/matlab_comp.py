from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from data.lm_coord_turn_example import get_specific_states_from_file
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import LmCoordTurn


def main():
    ts_fin = 500
    dt = 0.01
    qc = 0.01
    qw = 10
    Q = np.array(
        [
            [qc * dt ** 3 / 3, 0, qc * dt ** 2 / 2, 0, 0],
            [0, qc * dt ** 3 / 3, 0, qc * dt ** 2 / 2, 0],
            [qc * dt ** 2 / 2, 0, qc * dt, 0, 0],
            [0, qc * dt ** 2 / 2, 0, qc * dt, 0],
            [0, 0, 0, 0, dt * qw],
        ]
    )

    motion_model = LmCoordTurn(dt, Q)

    sens_pos_1 = np.array([-1.5, 0.5])
    sens_pos_2 = np.array([1, 1])
    sensors = np.row_stack((sens_pos_1, sens_pos_2))
    std = 0.5
    R = std ** 2 * np.eye(2)
    meas_model = MultiSensorRange(sensors, R)

    m1 = np.array([0, 0, 1, 0, 0])
    P1 = np.diag([0.1, 0.1, 1, 1, 1])
    X, Z, ss_xf = get_specific_states_from_file(Path.cwd())
    X, Z, ss_xf = X[:ts_fin, :], Z[:ts_fin, :], ss_xf[:ts_fin, :]
    MMS, PPS, MM, PP = basic_eks(
        Z, m1, P1, Q, R, motion_model.mapping, motion_model.jacobian, meas_model.mapping, meas_model.jacobian
    )
    assert np.allclose(ss_xf, MM)
    print(ss_xf.shape, MM.shape)
    vis(X, Z, MM, ss_xf)


def vis(states, meas, xf, ss_xf):
    _, ax = plt.subplots()
    ax.plot(states[:, 0], states[:, 1], label="true")
    ax.plot(ss_xf[:, 0], ss_xf[:, 1], label="matlab")
    ax.plot(xf[:, 0], xf[:, 1], label="python")
    ax.legend()
    plt.show()


def basic_eks(Z, m1, P1, Q, R, f_fun, df_fun, h_fun, dh_fun):
    m = m1
    P = P1
    D_x = m1.shape[0]
    ts_fin = Z.shape[0]
    MM = np.zeros((ts_fin, D_x))
    PP = np.zeros((ts_fin, D_x, D_x))

    for k in range(ts_fin):
        if k > 0:
            f = f_fun(m)
            F = df_fun(m)
            m = f
            P = F @ P @ F.T + Q

        h = h_fun(m)
        H = dh_fun(m)

        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        m = m + K @ (Z[k, :] - h)
        P = P - K @ S @ K.T
        MM[k, :] = m
        PP[k, :, :] = P

    ms = MM[-1, :]
    Ps = PP[-1, :, :]
    MMS = np.empty(MM.shape)
    PPS = np.empty(PP.shape)
    MMS[-1, :] = ms
    PPS[-1, :] = Ps
    for k in range(ts_fin - 2, 0, -1):
        m = MM[k, :]
        P = PP[k, :, :]

        f = f_fun(m)
        F = df_fun(m)
        mp = f
        Pp = F @ P @ F.T + Q
        Ck = P @ F.T @ np.linalg.inv(Pp)
        ms = m + Ck @ (ms - mp)
        Ps = P + Ck @ (Ps - Pp) @ Ck.T
        MMS[k, :] = ms
        PPS[k, :, :] = Ps

    return MMS, PPS, MM, PP


if __name__ == "__main__":
    main()

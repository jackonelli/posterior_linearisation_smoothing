from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from data.lm_ieks_paper.coord_turn_example import get_specific_states_from_file, Type
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
    X, Z, ss_xf, ss_xs = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.Extended)
    X, Z, ss_xf, ss_xs = X[:ts_fin, :], Z[:ts_fin, :], ss_xf[:ts_fin, :], ss_xs[:ts_fin, :]
    MMS, PPS, MM, PP = basic_eks(
        Z,
        m1,
        P1,
        Q,
        R,
        motion_model.mapping,
        motion_model.jacobian,
        meas_model.mapping,
        meas_model.jacobian,
    )
    assert np.allclose(ss_xf, MM)
    assert np.allclose(ss_xs, MMS)
    MM, PP, MMS, PPS = gn_eks(
        Z,
        m1,
        P1,
        Q,
        R,
        motion_model.mapping,
        motion_model.jacobian,
        meas_model.mapping,
        meas_model.jacobian,
        10,
        np.zeros((ts_fin, m1.shape[0])),
    )
    _, _, ss_ixf, ss_ixs = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.GN)
    # vis(X, Z, loc=MM, ss=ss_ixf)
    assert np.allclose(ss_ixf, MM)
    # vis(X, Z, MMS, ss_ixs)
    assert np.allclose(ss_ixs, MMS)


def vis(states, meas, loc, ss):
    _, ax = plt.subplots()
    ax.plot(states[:, 0], states[:, 1], label="true")
    # ax.plot(ss[:, 0], ss[:, 1], label="matlab")
    ax.plot(loc[:, 0], loc[:, 1], label="GN-IEKS")
    ax.legend()
    plt.show()


def gn_eks(measurements, prior_mean, prior_cov, Q, R, f_fun, df_fun, h_fun, dh_fun, niter, MN0):
    MN = MN0
    # JJ = np.zeros((niter, 1))
    # J = ss_cost(MN,Z,m1,P1,Q,R,f_fun,h_fun)
    D_x = prior_mean.shape[0]
    ts_fin = measurements.shape[0]
    for iter_ in range(1, niter + 1):
        print("Iter: ", iter_)
        m = prior_mean
        P = prior_cov
        xf = np.zeros((ts_fin, D_x))
        Pf = np.zeros((ts_fin, D_x, D_x))
        for k in range(ts_fin):
            if k > 0:
                # Fix the nominal distribution for prediction
                mn = MN[k - 1, :]

                F = df_fun(mn)
                f = f_fun(mn) + F @ (m - mn)

                m = f
                P = F @ P @ F.T + Q
            mn = MN[k, :]

            H = dh_fun(mn)
            h = h_fun(mn) + H @ (m - mn)

            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            m = m + K @ (measurements[k, :] - h)
            P = P - K @ S @ K.T

            xf[k, :] = m
            Pf[k, :, :] = P
        ms_K = xf[-1, :]
        Ps_K = Pf[-1, :, :]

        xs = np.empty(xf.shape)
        Ps = np.empty(Pf.shape)
        xs[-1, :] = ms_K
        Ps[-1, :, :] = Ps_K

        for k in np.flip(np.arange(0, ts_fin - 1)):
            m = xf[k, :]
            P = Pf[k, :, :]

            mn = MN[k, :]

            F = df_fun(mn)
            f = f_fun(mn) + F @ (m - mn)

            mp = f
            Pp = F @ P @ F.T + Q
            Ck = P @ F.T @ np.linalg.inv(Pp)

            ms_K = m + Ck @ (ms_K - mp)
            Ps_K = P + Ck @ (Ps_K - Pp) @ Ck.T
            xs[k, :] = ms_K
            Ps[k, :, :] = Ps_K

        # % No line search
        MN = xs.copy()
    return xf, Pf, xs, Ps


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
    for k in np.flip(np.arange(0, ts_fin - 1)):
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

    print("last k: ", k)

    return MMS, PPS, MM, PP


if __name__ == "__main__":
    main()

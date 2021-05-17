"""Python port of Simo Särkkäs matlab impl

Used for debugging
"""
import numpy as np
from src.cost import ss_cost


def ls_ieks(measurements, prior_mean, prior_cov, Q, R, f_fun, df_fun, h_fun, dh_fun, niter, MN0, ngrid):
    MN = MN0
    # JJ = np.zeros((niter, 1))
    # J = ss_cost(MN,Z,m1,P1,Q,R,f_fun,h_fun)
    J = ss_cost(MN, measurements, prior_mean, prior_cov, Q, R, f_fun, h_fun)
    D_x = prior_mean.shape[0]
    ts_fin = measurements.shape[0]
    for iter_ in range(1, niter + 1):
        print(MN.sum())
        # print("Matlab iter: ", iter_)
        # print(MN.sum())
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

        # Grid-based line search
        DMN = xs - MN

        Jp_list = np.zeros((ngrid, 1))
        MNp_list = list()

        for j in range(1, ngrid + 1):
            j_ind = j - 1
            MNp_list.append(MN + (j / ngrid) * DMN)
            Jp_list[j_ind] = ss_cost(MNp_list[j_ind], measurements, prior_mean, prior_cov, Q, R, f_fun, h_fun)

        # [min_Jp,min_j] = min(Jp_list)
        min_j = np.argmin(Jp_list)
        min_Jp = Jp_list[min_j, 0]

        if min_Jp > J:
            print("Grid search did not decrease, defaulting to plain GN.")
            MN = MNp_list[-1]
            J = Jp_list[-1]
            alpha = 1
        else:
            J = min_Jp
            MN = MNp_list[min_j]
            alpha = min_j / ngrid
        # m = NaN

        # % No line search
        # MN = xs.copy()
    return xf, Pf, xs, Ps


from pathlib import Path
from functools import partial
import numpy as np
from src.smoother.ext.ls_ieks import LsIeks
from src.line_search import GridSearch
from src.cost import analytical_smoothing_cost
from src.models.range_bearing import MultiSensorBearings
from src.models.coord_turn import CoordTurn
from data.lm_ieks_paper.coord_turn_example import get_specific_states_from_file, Type
from src.smoother.ext.ls_ieks import LsIeks


def main():
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
    motion_model = CoordTurn(dt, Q)

    sens_pos_1 = np.array([-1.5, 0.5])
    sens_pos_2 = np.array([1, 1])
    sensors = np.row_stack((sens_pos_1, sens_pos_2))
    std = 0.5
    R = std ** 2 * np.eye(2)
    meas_model = MultiSensorBearings(sensors, R)

    prior_mean = np.array([0, 0, 1, 0, 0])
    prior_cov = np.diag([0.1, 0.1, 1, 1, 1])
    # Weird params
    # prior_mean = np.array([-1, -1, 0, 0, 0])
    # prior_cov = np.eye(5)

    num_iter = 2
    _, measurements, ss_mf, ss_ms = get_specific_states_from_file(
        Path.cwd() / "data/lm_ieks_paper", Type.LineSearch, num_iter
    )
    measurements = measurements[:, 2:]

    K = measurements.shape[0]
    init_traj = (np.zeros((K, prior_mean.shape[0])), None)

    _, _, ms_ss, Ps_ss = ls_ieks(
        measurements,
        prior_mean,
        prior_cov,
        Q,
        R,
        motion_model.mapping,
        partial(motion_model.jacobian, _time_step=0),
        meas_model.mapping,
        partial(meas_model.jacobian, time_step=0),
        num_iter,
        init_traj[0],
        10,
    )

    cost_fn = partial(
        analytical_smoothing_cost,
        measurements=measurements,
        m_1_0=prior_mean,
        P_1_0=prior_cov,
        motion_model=motion_model,
        meas_model=meas_model,
    )

    ls_method = GridSearch(cost_fn, 10)
    smoother = LsIeks(motion_model, meas_model, num_iter, ls_method)
    _, _, ms, Ps, costs = smoother.filter_and_smooth_with_init_traj(
        measurements, prior_mean, prior_cov, init_traj, 1, cost_fn
    )
    # print(ms_ss.sum(), ms.sum())


if __name__ == "__main__":
    main()

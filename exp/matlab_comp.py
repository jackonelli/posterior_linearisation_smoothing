from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from data.lm_ieks_paper.coord_turn_example import get_specific_states_from_file, Type
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import LmCoordTurn
from src.smoother.ext.cost import cost
from src.ss.gn import gn_ieks
from src.ss.eks import basic_eks
from src.ss.lm import lm_ieks


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
    num_iter = 1
    # X, Z, ss_xf, ss_xs = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.Extended, None)
    # X, Z, ss_xf, ss_xs = X[:ts_fin, :], Z[:ts_fin, :], ss_xf[:ts_fin, :], ss_xs[:ts_fin, :]
    # MMS, PPS, MM, PP = basic_eks(
    #     Z,
    #     m1,
    #     P1,
    #     Q,
    #     R,
    #     motion_model.mapping,
    #     motion_model.jacobian,
    #     meas_model.mapping,
    #     meas_model.jacobian,
    # )
    # assert np.allclose(ss_xf, MM)
    # assert np.allclose(ss_xs, MMS)
    X, Z, ss_imf, ss_ims = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.GN, num_iter)
    # vis(X, Z, loc=MM, ss=ss_ixf)
    print("Cost: ", cost(ss_ims, Z, m1, P1, motion_model, meas_model))
    MM, PP, MMS, PPS = lm_ieks(
        Z,
        m1,
        P1,
        Q,
        R,
        motion_model.mapping,
        motion_model.jacobian,
        meas_model.mapping,
        meas_model.jacobian,
        num_iter,
        np.zeros((ts_fin, m1.shape[0])),
    )
    assert np.allclose(ss_imf, MM)
    # vis(X, Z, MMS, ss_ixs)
    assert np.allclose(ss_ims, MMS)


def vis(states, meas, loc, ss):
    _, ax = plt.subplots()
    ax.plot(states[:, 0], states[:, 1], label="true")
    # ax.plot(ss[:, 0], ss[:, 1], label="matlab")
    ax.plot(loc[:, 0], loc[:, 1], label="GN-IEKS")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from data.lm_ieks_paper.coord_turn_example import get_specific_states_from_file, Type
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import LmCoordTurn
from src.smoother.ext.cost import cost, ss_cost
from src.ss.lm import lm_ieks
from src.ss.hybrid_lm import hybrid_lm_ieks
import src.visualization as vis


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

    motion_model = LmCoordTurn(dt, Q)

    sens_pos_1 = np.array([-1.5, 0.5])
    sens_pos_2 = np.array([1, 1])
    sensors = np.row_stack((sens_pos_1, sens_pos_2))
    std = 0.5
    R = std ** 2 * np.eye(2)
    meas_model = MultiSensorRange(sensors, R)

    m1 = np.array([0, 0, 1, 0, 0])
    P1 = np.diag([0.1, 0.1, 1, 1, 1])
    num_iter = 2
    # X, Z, ss_xf, ss_xs = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.Extended, None)
    # X, Z, ss_xf, ss_xs = X[:ts_fin, :], Z[:ts_fin, :], ss_xf[:ts_fin, :], ss_xs[:ts_fin, :]

    X, Z, _, _ = get_specific_states_from_file(Path.cwd() / "data/lm_ieks_paper", Type.LM, 1)
    K = X.shape[0]
    print("Matlab version")
    mf_ss, Pf_ss, ms_ss, Ps_ss = lm_ieks(
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
        np.zeros((K, m1.shape[0])),
    )

    mf_h, Pf_h, ms_h, Ps_h = hybrid_lm_ieks(
        Z,
        m1,
        P1,
        motion_model,
        meas_model,
        num_iter,
        np.zeros((K, m1.shape[0])),
    )

    print("Py version")
    lambda_ = 1e-2
    nu = 10
    assert np.allclose(mf_ss, mf_h)
    assert np.allclose(ms_ss, ms_h)


if __name__ == "__main__":
    main()

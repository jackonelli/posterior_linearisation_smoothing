"""Experiment: Arctan measurement model

Reproducing the experiment in the paper:

"Damped Posterior Linearization Filter"
"""

import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src import visualization as vis
from src.filter.ekf import Ekf
from src.smoother.ext.eks import Eks
from src.smoother.ext.ieks import Ieks
from src.smoother.ext.lm_ieks import LmIeks
from src.smoother.ipls import SigmaPointIpls
from src.utils import setup_logger
from src.models.range_bearing import MultiSensorRange
from src.models.coord_turn import CoordTurn
from data.lm_ieks_paper.coord_turn_example import Type, get_specific_states_from_file
from exp.matlab_comp import gn_ieks, basic_eks


def main():
    log = logging.getLogger(__name__)
    experiment_name = "damped_plf_arctan"
    setup_logger(f"logs/{experiment_name}.log", logging.INFO)
    log.info(f"Running experiment: {experiment_name}")
    seed = 2
    np.random.seed(seed)


if __name__ == "__main__":
    main()

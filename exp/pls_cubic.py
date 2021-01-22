"""Experiment: Non-stationary growth with cubic measurement model.

Reproducing the experiment in the paper:

"Iterated Posterior Linearization Smoother"
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
from src.models.coord_turn import LmCoordTurn
from data.lm_ieks_paper.coord_turn_example import Type, get_specific_states_from_file
from exp.matlab_comp import gn_ieks, basic_eks

from src.smoother.ext.cost import cost


def main():
    log = logging.getLogger(__name__)
    experiment_name = "pls_cubic"
    setup_logger(f"logs/{experiment_name}.log", logging.INFO)
    log.info(f"Running experiment: {experiment_name}")
    seed = 2
    np.random.seed(seed)


if __name__ == "__main__":
    main()

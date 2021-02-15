"""Utils"""
import sys
import logging
from datetime import datetime
from typing import Union, Optional
from pathlib import Path
import numpy as np

LOG_FORMAT = "%(asctime)-15s %(levelname)-5s %(name)-15s - %(message)s"


def setup_logger(log_path: Union[str, Path], log_level: Union[str, int], fmt: Optional[str] = LOG_FORMAT):
    """Setup for a logger instance.
    Args:
        log_path: full path
        log_level:
        fmt: message format
    """
    logger = logging.getLogger()
    formatter = logging.Formatter(fmt=fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.setLevel(log_level)
    logger.handlers = []
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    log_path = Path(log_path)
    directory = log_path.parent
    directory.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(str(log_path))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Matplotlib is _very_ noisy at DEBUG level.
    # Set to WARNING for good measure.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logger.info("Log at {}".format(log_path))


def log_name(name: str) -> str:
    """Generate log name"""
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    return "{}_{}.log".format(name, timestamp)


def make_pos_def(x, eps=0.1):
    u, s, vh = np.linalg.svd(x, hermitian=True)
    neg_sing_vals = s < 0
    s_hat = s * np.logical_not(neg_sing_vals) + eps * neg_sing_vals
    return np.dot(u, np.dot(np.diag(s_hat), vh)), s


def tikz_err_bar_tab_to_file(xs, ys, errs, file_: Path):
    with open(file_, "w") as data_file:
        data_file.write(tikz_err_bar_tab_format(xs, ys, errs))


def tikz_err_bar_tab_format(xs, ys, err):
    header = "x y err"
    data = [f"{x} {y} {err}" for x, y, err in zip(xs, ys, err)]
    data = "\n".join(data)
    return f"{header}\n{data}"


def tikz_2d_tab_to_file(data, dir_: Path):
    for label, vals in data:
        with open(dir_ / f"{label.lower()}.data", "w") as data_file:
            data_file.write(tikz_2d_tab_format(vals[:, 0], vals[:, 1]))


def tikz_2d_tab_format(xs, ys):
    header = "x y"
    data = [f"{x} {y}" for x, y in zip(xs, ys)]
    data = "\n".join(data)
    return f"{header}\n{data}"

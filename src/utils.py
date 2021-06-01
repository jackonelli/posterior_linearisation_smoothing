"""Utils"""
import sys
import logging
from datetime import datetime
from typing import Union, Optional
from pathlib import Path
import numpy as np
from src.analytics import mc_stats

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


def tikz_err_bar_tab_format(xs, ys, err):
    print(err)
    header = "x y err"
    data = [f"{x} {y} {err}" for x, y, err in zip(xs, ys, err)]
    data = "\n".join(data)
    return f"{header}\n{data}"


def tikz_2d_tab_format(xs, ys):
    header = "x y"
    data = [f"{x} {y}" for x, y in zip(xs, ys)]
    data = "\n".join(data)
    return f"{header}\n{data}"


def tikz_2d_traj(dir_, traj, label):
    tikz_dir = dir_ / "traj"
    tikz_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(tikz_dir / f"{label.lower()}.data", traj, header="x y", comments="")


def tikz_1d_tab_format(ys):
    xs = np.arange(1, len(ys) + 1)
    header = "x y"
    data = [f"{x} {y}" for x, y in zip(xs, ys)]
    data = "\n".join(data)
    return f"{header}\n{data}"


def tikz_stats(dir_, name, stats):
    metric_dir = dir_ / name
    metric_dir.mkdir(parents=True, exist_ok=True)
    num_iter = stats[0][0].shape[1]
    iter_range = np.arange(1, num_iter + 1)
    stats = [(mc_stats(stat_), label) for stat_, label in stats]
    for (mean, err), label in stats:
        data = np.column_stack((iter_range, mean, err))
        np.savetxt(metric_dir / f"{label.lower()}.data", data, header="x y err", comments="")


def save_stats(res_dir: Path, name: str, stats):
    (res_dir / name.lower()).mkdir(parents=True, exist_ok=True)
    for stat, label in stats:
        np.savetxt(res_dir / name.lower() / f"{label.lower()}.csv", stat)

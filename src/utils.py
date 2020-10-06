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

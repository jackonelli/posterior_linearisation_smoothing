import argparse
import numpy as np
from pathlib import Path
from src.utils import tikz_stats
from src.visualization import plot_scalar_metric_err_bar


def main():
    args = parse_args()
    print(args.dir)
    smoothers = [
        "ieks",
        "lm-ieks",
        "ls-ieks",
        "ipls",
        "lm-ipls",
        "ls-ipls",
    ]
    rmse_stats = [(np.loadtxt(args.dir / f"rmse/{label}.csv"), label.upper()) for label in smoothers]
    nees_stats = [(np.loadtxt(args.dir / f"nees/{label}.csv"), label.upper()) for label in smoothers]
    plot_scalar_metric_err_bar(rmse_stats, "RMSE")
    plot_scalar_metric_err_bar(nees_stats, "NEES")
    tikz_stats(Path.cwd() / "tikz", "rmse", rmse_stats)
    tikz_stats(Path.cwd() / "tikz", "nees", nees_stats)


def parse_args():
    parser = argparse.ArgumentParser(description="Parse results from experiments")
    parser.add_argument("--dir", type=Path, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    main()

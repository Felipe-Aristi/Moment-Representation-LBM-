#!/usr/bin/env python3

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


VALIDATION_DIRECTORY = Path(__file__).resolve().parent
POST_DIRECTORY = VALIDATION_DIRECTORY.parents[1]
PLOTS_DIRECTORY = VALIDATION_DIRECTORY / "plots"

sys.path.insert(0, str(POST_DIRECTORY))
sys.path.insert(0, str(VALIDATION_DIRECTORY))

from prettyPlot import pretty_plot
from post.channelflow.validation.statistics_io import load_enstrophy
from benchmark import REFERENCE_LABEL, load_benchmark


def enstrophy_limits(enstrophy_star: np.ndarray) -> tuple[float, float]:
    values = np.asarray(enstrophy_star, dtype=np.float64)

    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Enstrophy history must contain finite data")

    lower_data = float(np.min(values))
    upper_data = float(np.max(values))

    span = upper_data - lower_data

    if span <= np.finfo(np.float64).eps * max(
        abs(lower_data),
        abs(upper_data),
        1.0,
    ):
        padding = max(0.05 * abs(upper_data), 1.0e-4)
    else:
        padding = 0.06 * span

    lower_limit = max(0.0, lower_data - padding)
    upper_limit = upper_data + padding

    return lower_limit, upper_limit


def plot_enstrophy() -> Path:
    from mesh_plots import plot_mesh_comparison

    return plot_mesh_comparison("enstrophy")


def main() -> None:
    output_path = plot_enstrophy()

    print(
        f"Saved Taylor-Green enstrophy plot to {output_path}"
    )


if __name__ == "__main__":
    main()

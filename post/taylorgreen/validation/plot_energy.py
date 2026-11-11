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
from post.channelflow.validation.statistics_io import load_kinetic_energy
from benchmark import REFERENCE_LABEL, load_benchmark


INITIAL_KINETIC_ENERGY_STAR = 0.125


def kinetic_energy_limits(kinetic_energy_star: np.ndarray) -> tuple[float, float]:
    values = np.asarray(kinetic_energy_star, dtype=np.float64)

    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Kinetic-energy history must contain finite data")

    lower_data = min(float(np.min(values)), INITIAL_KINETIC_ENERGY_STAR)
    upper_data = max(float(np.max(values)), INITIAL_KINETIC_ENERGY_STAR)
    span = upper_data - lower_data

    if span <= np.finfo(np.float64).eps * max(abs(lower_data), abs(upper_data), 1.0):
        padding = max(0.05 * abs(INITIAL_KINETIC_ENERGY_STAR), 1.0e-4)
    else:
        padding = 0.06 * span

    lower_limit = lower_data - padding

    if lower_data >= 0.0:
        lower_limit = max(0.0, lower_limit)

    return lower_limit, upper_data + padding


def plot_kinetic_energy() -> Path:
    from mesh_plots import plot_mesh_comparison

    return plot_mesh_comparison("energy")


def main() -> None:
    output_path = plot_kinetic_energy()
    print(f"Saved Taylor-Green kinetic-energy plot to {output_path}")


if __name__ == "__main__":
    main()

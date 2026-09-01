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
from statistics_io import load_kinetic_energy


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
    history = load_kinetic_energy()
    y_limits = kinetic_energy_limits(history.KStar)

    fig, ax, _ = pretty_plot(
        xLim=(0.0, 20.0),
        yLim=y_limits,
        nxTicks=6,
        nyTicks=6,
        xLabel=r"$t^*$",
        yLabel=r"$K^*$",
        xTickFormat=0,
        yTickFormat=3,
        useGrid=True,
        plotAspectRatio=(1.5, 1.0, 1.0),
    )

    ax.plot(
        history.tStar,
        history.KStar,
        color="#0072B2",
        linestyle="-",
        linewidth=1.25,
        label="MR-RR",
    )
    ax.legend(frameon=False, loc="best")

    PLOTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIRECTORY / "kinetic_energy.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main() -> None:
    output_path = plot_kinetic_energy()
    print(f"Saved Taylor-Green kinetic-energy plot to {output_path}")


if __name__ == "__main__":
    main()

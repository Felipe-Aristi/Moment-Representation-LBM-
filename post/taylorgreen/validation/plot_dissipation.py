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
from benchmark import REFERENCE_LABEL, load_benchmark
from post.channelflow.validation.statistics_io import (
    load_kinetic_energy,
    load_enstrophy,
)


def compute_dissipation():
    energy = load_kinetic_energy()
    enstrophy = load_enstrophy()

    if not np.array_equal(energy.step, enstrophy.step):
        raise ValueError(
            "Kinetic-energy and enstrophy histories "
            "must use identical sample steps"
        )

    if not np.allclose(
        energy.tStar,
        enstrophy.tStar,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "Kinetic-energy and enstrophy histories "
            "must use identical tStar values"
        )

    t_star = energy.tStar
    k_star = energy.KStar
    zeta_star = enstrophy.ZetaStar

    reynolds = float(energy.metadata["Re"])

    epsilon_energy = -np.gradient(
        k_star,
        t_star,
        edge_order=2,
    )

    epsilon_enstrophy = (
        2.0 * zeta_star / reynolds
    )

    if not np.all(np.isfinite(epsilon_energy)):
        raise ValueError(
            "Energy-based dissipation contains NaN or Inf"
        )

    if not np.all(np.isfinite(epsilon_enstrophy)):
        raise ValueError(
            "Enstrophy-based dissipation contains NaN or Inf"
        )

    return (
        t_star,
        epsilon_energy,
        epsilon_enstrophy,
    )


def dissipation_limits(
    epsilon_energy: np.ndarray,
    epsilon_enstrophy: np.ndarray,
) -> tuple[float, float]:
    values = np.concatenate(
        (
            np.asarray(epsilon_energy, dtype=np.float64),
            np.asarray(epsilon_enstrophy, dtype=np.float64),
        )
    )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Dissipation histories must contain finite data"
        )

    lower = min(0.0, float(np.min(values)))
    upper = float(np.max(values))

    span = upper - lower

    if span <= np.finfo(np.float64).eps:
        padding = 1.0e-4
    else:
        padding = 0.06 * span

    return (
        lower - padding,
        upper + padding,
    )


def plot_dissipation() -> tuple[Path, Path]:
    from mesh_plots import plot_mesh_comparison

    return (
        plot_mesh_comparison("dissipation_energy"),
        plot_mesh_comparison("dissipation_enstrophy"),
    )


def main() -> None:
    for output_path in plot_dissipation():
        print(f"Saved Taylor-Green dissipation plot to {output_path}")


if __name__ == "__main__":
    main()

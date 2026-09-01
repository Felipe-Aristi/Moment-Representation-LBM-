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
from statistics_io import load_enstrophy


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
    history = load_enstrophy()

    y_limits = enstrophy_limits(history.ZetaStar)

    fig, ax, _ = pretty_plot(
        xLim=(0.0, 20.0),
        yLim=y_limits,
        nxTicks=6,
        nyTicks=6,
        xLabel=r"$t^*$",
        yLabel=r"$\zeta^*$",
        xTickFormat=0,
        yTickFormat=2,
        useGrid=True,
        plotAspectRatio=(1.5, 1.0, 1.0),
    )

    ax.plot(
        history.tStar,
        history.ZetaStar,
        color="#0072B2",
        linestyle="-",
        linewidth=1.25,
        label="MR-RR",
    )

    peak_index = int(np.argmax(history.ZetaStar))

    ax.plot(
        history.tStar[peak_index],
        history.ZetaStar[peak_index],
        marker="o",
        markersize=4.0,
        color="#0072B2",
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    PLOTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = PLOTS_DIRECTORY / "enstrophy.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Peak enstrophy: "
        f"tStar={history.tStar[peak_index]:.17g}, "
        f"ZetaStar={history.ZetaStar[peak_index]:.17g}"
    )

    return output_path


def main() -> None:
    output_path = plot_enstrophy()

    print(
        f"Saved Taylor-Green enstrophy plot to {output_path}"
    )


if __name__ == "__main__":
    main()
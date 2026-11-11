
#!/usr/bin/env python3
"""Channel-flow kinetic-energy plots matching the MR-LBM jet validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator

VALIDATION_DIR = Path(__file__).resolve().parent
POST_DIR = VALIDATION_DIR.parents[1]

sys.path.insert(0, str(POST_DIR))

from prettyPlot import pretty_plot
from exportAndCrop import export_and_crop

RED = "#E03A3E"
BLUE = "#009DDC"
GREEN = "#61BB46"


# ============================================================
# Find channel energy data
# ============================================================

def find_data(folder: Path | None) -> Path:

    if folder is not None:
        path = folder.expanduser().resolve()

        if not (path / "energy_history.csv").is_file():
            raise FileNotFoundError(
                f"Missing energy_history.csv in {path}"
            )

        return path

    files = list(
        (VALIDATION_DIR / "data").glob(
            "RETAU*_*/energy_history.csv"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No channel energy_history.csv found under validation/data"
        )

    return max(
        files,
        key=lambda p: p.stat().st_mtime
    ).parent


# ============================================================
# Read kinetic-energy history
# ============================================================

def load_energy(path: Path) -> np.ndarray:

    history = np.atleast_1d(
        np.genfromtxt(
            path,
            delimiter=",",
            names=True,
            dtype=np.float64,
        )
    )

    required = (
        "step",
        "t_star",
        "kinetic_energy_plus",
        "running_mean_plus",
        "relative_change",
    )

    if history.size < 2 or history.dtype.names is None:
        raise ValueError(
            "Energy history must contain at least two samples"
        )

    if any(name not in history.dtype.names for name in required):
        raise ValueError(
            f"Energy CSV is missing one of: {required}"
        )

    if (
        not np.all(np.isfinite(history["step"]))
        or np.any(np.diff(history["step"]) <= 0)
    ):
        raise ValueError(
            "Energy sample steps must increase strictly"
        )

    for name in (
        "t_star",
        "kinetic_energy_plus",
        "running_mean_plus",
    ):
        if not np.all(np.isfinite(history[name])):
            raise ValueError(
                f"Non-finite values in {name}"
            )

    if np.any(np.diff(history["t_star"]) <= 0):
        raise ValueError(
            "Dimensionless times must increase strictly"
        )

    if (
        np.any(history["kinetic_energy_plus"] < 0)
        or np.any(history["running_mean_plus"] < 0)
    ):
        raise ValueError(
            "Kinetic energy cannot be negative"
        )

    return history


# ============================================================
# Calculate energy axis limits
# ============================================================

def energy_limits(
    *arrays: np.ndarray,
) -> tuple[float, float]:

    values = np.concatenate([
        np.asarray(a, dtype=np.float64)
        for a in arrays
    ])

    low = float(np.min(values))
    high = float(np.max(values))

    margin = (
        0.06 * (high - low)
        if high > low
        else max(0.05 * max(high, 1.0), 1e-4)
    )

    return max(0.0, low - margin), high + margin


# ============================================================
# Create plot with original jet aesthetics
# ============================================================

def make_axis(
    t: np.ndarray,
    ylim: tuple[float, float],
    ylabel: str,
):

    xmax = max(float(t[-1]), 1e-9)

    fig, ax, _ = pretty_plot(
        xLim=(0.0, xmax),
        yLim=ylim,

        xLabel=r"$t^*=t u_\tau/H$",
        yLabel=ylabel,

        xTickFormat=2,
        yTickFormat=2,

        nxTicks=7,
        nyTicks=7,

        plotAspectRatio=(1, 1, 1),

        useGrid=False,
        useColorBar=False,

        paperPoints=612,
        marginPoints=54,
        textWidth=1,

        boxMarginScale=0.085,

        fontSize=14,
        dpi=300,
    )

    ax.yaxis.set_label_coords(-0.14, 0.5)

    return fig, ax


# ============================================================
# Generate figures
# ============================================================

def generate_plots(
    data_dir: Path | None,
    output_dir: Path,
    all_jet_figures: bool,
):

    folder = find_data(data_dir)

    h = load_energy(
        folder / "energy_history.csv"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Energy variables
    # ========================================================

    t = h["t_star"]

    total = h["kinetic_energy_plus"]
    mean = h["running_mean_plus"]

    limits = energy_limits(total, mean)

    # ========================================================
    # Figure 1: instantaneous kinetic energy
    # ========================================================

    fig, ax = make_axis(
        t,
        limits,
        r"$K^+$",
    )

    ax.plot(
        t,
        total,
        color=RED,
        linewidth=2,
        label="Instantaneous",
    )

    ax.legend(
        frameon=False,
        fontsize=11,
    )

    output_path = (
        output_dir / "kinetic_energy_total.png"
    )

    export_and_crop(fig, output_path)

    print(f"Saved: {output_path}")

    # ========================================================
    # Figure 2: running mean kinetic energy
    # ========================================================

    fig, ax = make_axis(
        t,
        limits,
        r"$\overline{K^+}^{\,(t)}$",
    )

    ax.plot(
        t,
        mean,
        color=BLUE,
        linewidth=2,
        label="Running mean",
    )

    ax.legend(
        frameon=False,
        fontsize=11,
    )

    output_path = (
        output_dir / "kinetic_energy_avg.png"
    )

    export_and_crop(fig, output_path)

    print(f"Saved: {output_path}")

    # ========================================================
    # Optional: the other two jet diagnostics
    # ========================================================

    if all_jet_figures:

        # ====================================================
        # Figure 3: total energy vs running mean
        # ====================================================

        fig, ax = make_axis(
            t,
            limits,
            r"$K^+$",
        )

        ax.plot(
            t,
            total,
            color=RED,
            linewidth=2,
            label="Instantaneous",
        )

        ax.plot(
            t,
            mean,
            color=BLUE,
            linewidth=2,
            linestyle="--",
            label="Running mean",
        )

        ax.legend(
            frameon=False,
            fontsize=11,
        )

        output_path = (
            output_dir / "kinetic_energy_total_vs_avg.png"
        )

        export_and_crop(fig, output_path)

        print(f"Saved: {output_path}")

        # ====================================================
        # Figure 4: relative energy variation
        # ====================================================

        relative = h["relative_change"]

        valid = (
            np.isfinite(relative)
            & (relative > 0)
        )

        if np.any(valid):

            ymin = max(
                float(np.min(relative[valid])) * 0.8,
                1e-15,
            )

            ymax = max(
                float(np.max(relative[valid])) * 1.2,
                ymin * 10,
            )

            fig, ax = make_axis(
                t,
                (ymin, ymax),
                r"$|K_n-K_{n-1}|/K_{n-1}$",
            )

            ax.set_yscale("log")

            ax.yaxis.set_major_locator(
                LogLocator(base=10)
            )

            ax.yaxis.set_major_formatter(
                LogFormatterMathtext(base=10)
            )

            ax.plot(
                t[valid],
                relative[valid],
                color=GREEN,
                linewidth=1.8,
            )

            output_path = (
                output_dir / "kinetic_energy_diff.png"
            )

            export_and_crop(fig, output_path)

            print(f"Saved: {output_path}")

        else:
            print(
                "Skipping relative-change graph: "
                "no finite positive changes"
            )

    # ========================================================
    # Terminal information
    # ========================================================

    print(f"Data: {folder}")

    print(
        f"Last step: {int(h['step'][-1])}; "
        f"last t*: {t[-1]:.6g}"
    )

    print(
        f"Last K+: {total[-1]:.8g}; "
        f"running mean K+: {mean[-1]:.8g}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Directory containing energy_history.csv",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=VALIDATION_DIR / "plots" / "results",
    )

    parser.add_argument(
        "--all-jet-figures",
        action="store_true",
        help=(
            "Also plot relative energy change "
            "and total vs running mean"
        ),
    )

    args = parser.parse_args()

    try:
        generate_plots(
            args.data,
            args.output,
            args.all_jet_figures,
        )

    except (OSError, ValueError, KeyError) as exc:
        parser.exit(
            1,
            f"plot_energy.py: {exc}\n",
        )


if __name__ == "__main__":
    main()

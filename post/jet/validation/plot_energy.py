#!/usr/bin/env python3
"""Jet domain kinetic energy, cumulative running mean, and relative changes.

Uses energy_history.csv from cases/jet/energy_statistics.cuh.
Unlike the old jet code's binary-only histories, the CSV includes the exact
completed step for each sample, including a possible final partial interval.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "Re5000" / "mean_profiles"


def load_energy_history(data_dir: Path) -> np.ndarray:
    path = data_dir / "energy_history.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing energy history: {path}")
    history = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64))
    columns = ("step", "t_star", "energy_total", "energy_running_mean",
               "relative_change", "energy_star", "energy_running_mean_star")
    if history.size == 0 or history.dtype.names is None or any(k not in history.dtype.names for k in columns):
        raise ValueError("Energy history is empty or columns do not match the writer")
    if not np.all(np.isfinite(history["t_star"])) or not np.all(np.isfinite(history["step"])):
        raise ValueError("Non-finite time coordinates")
    if np.any(np.diff(history["step"]) <= 0) or np.any(np.diff(history["t_star"]) <= 0):
        raise ValueError("Energy sample times must increase strictly")
    for col in ("energy_total", "energy_running_mean", "energy_star", "energy_running_mean_star"):
        if not np.all(np.isfinite(history[col])) or np.any(history[col] < 0):
            raise ValueError(f"Invalid data in {col}")
    return history


def save_plot(path: Path, title: str, xlabel: str, ylabel: str, x: np.ndarray,
              curves: list[tuple[np.ndarray, str]], *, log_y: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for y, label in curves:
        valid = np.isfinite(x) & np.isfinite(y) & ((y > 0) if log_y else np.ones(y.shape, dtype=bool))
        if np.any(valid):
            ax.plot(x[valid], y[valid], lw=1.7, label=label)
    if not ax.lines:
        plt.close(fig)
        print(f"Skipping {path.name}: not enough finite positive data")
        return
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def generate_energy_plots(data_dir: Path, output_dir: Path) -> None:
    h = load_energy_history(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t = h["t_star"]
    total = h["energy_star"]
    average = h["energy_running_mean_star"]
    relative = h["relative_change"]

    save_plot(output_dir / "kinetic_energy_total.png", "Domain kinetic energy",
              r"$t^* = t U_{jet}/D$", r"$E_K^*$", t,
              [(total, "MR-RR")])
    save_plot(output_dir / "kinetic_energy_avg.png", "Running mean of domain kinetic energy",
              r"$t^* = t U_{jet}/D$", r"$\overline{E_K^*}^{\,(t)}$", t,
              [(average, "Running mean")])
    save_plot(output_dir / "kinetic_energy_diff.png", "Relative energy change between samples",
              r"$t^* = t U_{jet}/D$", r"$|E_{K,n}-E_{K,n-1}|/E_{K,n-1}$", t,
              [(relative, "Relative change")], log_y=True)
    save_plot(output_dir / "kinetic_energy_total_vs_avg.png", "Domain kinetic energy and running mean",
              r"$t^* = t U_{jet}/D$", r"$E_K^*$", t,
              [(total, "Instantaneous"), (average, "Running mean")])

    valid_rel = relative[np.isfinite(relative)]
    print(f"Samples: {h.size}; last step: {int(h['step'][-1])}; last t*: {t[-1]:.8g}")
    print(f"Last E_K*: {total[-1]:.8g}; last running mean E_K*: {average[-1]:.8g}")
    if valid_rel.size:
        print(f"Last finite relative change: {valid_rel[-1]:.8g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Folder containing energy_history.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "plots",
                        help="Destination for kinetic-energy figures")
    args = parser.parse_args()
    generate_energy_plots(args.data, args.output)

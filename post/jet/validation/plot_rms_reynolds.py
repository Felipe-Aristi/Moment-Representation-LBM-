#!/usr/bin/env python3
"""MR-LBM jet RMS and Reynolds stress using original jet solver's plot style."""
from pathlib import Path
import argparse
import sys

import numpy as np

POST_DIR = Path(__file__).resolve().parents[2]
if str(POST_DIR) not in sys.path:
    sys.path.insert(0, str(POST_DIR))

from prettyPlot import pretty_plot
from exportAndCrop import export_and_crop
from statistics_io import DEFAULT_DATA_DIR, fit_virtual_origin, load_radial_profiles

red = "#E03A3E"
blue = "#009DDC"
green = "#61BB46"
purple = "#963D97"
yellow = "#FDB827"
orange = "#F5821F"
black = "#000000"
colors = [red, blue, green, purple, orange, yellow]
markers = ["o", "s", "^", "D", "v", "<", ">"]

# Same digitized Hussein et al. (1994) values used by the previous solver.
HUSSEIN_UY_RMS_X = np.array([
    0.000, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070,
    0.080, 0.090, 0.100, 0.110, 0.120, 0.130, 0.140, 0.150,
    0.160, 0.170, 0.180, 0.190, 0.200, 0.210, 0.220, 0.235,
])
HUSSEIN_UY_RMS = np.array([
    0.278, 0.282, 0.286, 0.286, 0.283, 0.276, 0.265, 0.252,
    0.235, 0.216, 0.196, 0.176, 0.155, 0.136, 0.118, 0.101,
    0.084, 0.069, 0.056, 0.045, 0.036, 0.029, 0.023, 0.018,
])
HUSSEIN_UR_RMS_X = np.array([
    0.000, 0.015, 0.030, 0.045, 0.060, 0.075, 0.090, 0.105,
    0.120, 0.135, 0.150, 0.165, 0.180, 0.195, 0.210, 0.225,
    0.240, 0.260, 0.280, 0.300,
])
HUSSEIN_UR_RMS = np.array([
    0.235, 0.224, 0.216, 0.207, 0.197, 0.185, 0.174, 0.162,
    0.148, 0.132, 0.114, 0.096, 0.079, 0.063, 0.049, 0.037,
    0.027, 0.017, 0.011, 0.007,
])
HUSSEIN_URUY_X = np.array([
    0.000, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070,
    0.080, 0.090, 0.105, 0.120, 0.135, 0.150, 0.170, 0.190,
    0.210, 0.230,
])
HUSSEIN_URUY = np.array([
    0.000, 0.006, 0.011, 0.016, 0.020, 0.022, 0.022, 0.021,
    0.019, 0.017, 0.014, 0.011, 0.009, 0.007, 0.0045, 0.0025,
    0.0010, 0.0002,
])


def make_pretty_rms_axis(xlim, ylim, ylabel):
    fig, ax, _ = pretty_plot(
        xLim=xlim,
        yLim=ylim,
        cLim=(-1, 1),
        plotAspectRatio=(1, 1, 1),
        xLabel=r"$r/(y-y_0)$",
        yLabel=ylabel,
        yScientificNotation=False,
        xTickFormat=2,
        yTickFormat=2,
        nxTicks=5,
        nyTicks=7,
        useColorBar=False,
        paperPoints=612,
        marginPoints=54,
        textWidth=1,
        boxMarginScale=0.085,
        yLabelAngle=90,
        useGrid=False,
        fontSize=14,
        dpi=300,
    )
    ax.yaxis.set_label_coords(-0.125, 0.5)
    return fig, ax


def plot_lbm_slices(ax, radius, values, count, uc, y0, diameter, slices):
    plotted = 0
    for j, slice_over_d in enumerate(slices):
        y_idx = int(round(slice_over_d * diameter))
        if y_idx <= 0 or y_idx >= values.shape[0] or not np.isfinite(uc[y_idx]) or uc[y_idx] <= 0.0:
            print(f"Skipping y/D={slice_over_d}: outside domain or invalid velocity.")
            continue
        denom = y_idx - y0
        if denom <= 0.0:
            print(f"Skipping y/D={slice_over_d}: y-y0 <= 0")
            continue
        valid = (count[y_idx, :] > 0) & np.isfinite(values[y_idx, :])
        if not np.any(valid):
            continue
        color = colors[j % len(colors)]
        marker = markers[j % len(markers)]
        label = rf"$y={slice_over_d:g}D$"
        ax.plot(radius[valid] / denom, values[y_idx, valid],
                ls="none", marker=marker, ms=3, color=color, label=label)
        plotted += 1
    return plotted


def plot_reference(ax, x, y):
    ax.plot(x, y, ls="none", marker="+", ms=6, mew=2,
            color=black, label="Hussein et al. (1994)")


def plot_rms_reynolds(data_dir, output_dir, fit_start=10.0, fit_end=15.0,
                       slices=(10, 11, 12, 13, 14, 15), stress_sign=1.0):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p = load_radial_profiles(data_dir)
    uy, count = p.uy, p.count
    diameter = float(p.metadata["D"])
    u_jet = float(p.metadata["jet_velocity"])
    ny, nr = uy.shape
    y = np.arange(ny, dtype=np.float64)
    radius = np.arange(nr, dtype=np.float64)
    uc = uy[:, 0]
    B, y0, _, _ = fit_virtual_origin(y, uc, diameter, u_jet, fit_start, fit_end)
    print(f"Using mean profile folder: {data_dir}")
    print(f"B = {B}")
    print(f"y0 = {y0}")

    with np.errstate(divide="ignore", invalid="ignore"):
        uy_rms_norm = p.uy_rms / uc[:, None]
        ur_rms_norm = p.ur_rms / uc[:, None]
        uruy_norm = stress_sign * p.reynolds_uruy / (uc[:, None] * uc[:, None])

    fig_uy, ax_uy = make_pretty_rms_axis(
        xlim=(0, 0.4), ylim=(0, 0.3), ylabel=r"$u_{y,rms}/U_{y,c}$")
    plot_lbm_slices(ax_uy, radius, uy_rms_norm, count, uc, y0, diameter, slices)
    plot_reference(ax_uy, HUSSEIN_UY_RMS_X, HUSSEIN_UY_RMS)
    ax_uy.legend(fontsize=10)
    export_and_crop(fig_uy, output_dir / "uy_rms_profile.png")

    fig_ur, ax_ur = make_pretty_rms_axis(
        xlim=(0, 0.4), ylim=(0, 0.3), ylabel=r"$u_{r,rms}/U_{y,c}$")
    plot_lbm_slices(ax_ur, radius, ur_rms_norm, count, uc, y0, diameter, slices)
    plot_reference(ax_ur, HUSSEIN_UR_RMS_X, HUSSEIN_UR_RMS)
    ax_ur.legend(fontsize=10)
    export_and_crop(fig_ur, output_dir / "ur_rms_profile.png")

    fig_stress, ax_stress = make_pretty_rms_axis(
        xlim=(0, 0.3), ylim=(0, 0.04),
        ylabel=r"$\langle u_r'u_y'\rangle/U_{y,c}^{2}$")
    plot_lbm_slices(ax_stress, radius, uruy_norm, count, uc, y0, diameter, slices)
    plot_reference(ax_stress, HUSSEIN_URUY_X, HUSSEIN_URUY)
    ax_stress.legend(fontsize=10)
    export_and_crop(fig_stress, output_dir / "uruy_reynolds_profile.png")


def main():
    parser = argparse.ArgumentParser(description="MR-LBM jet turbulence profiles, original plot style")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "plots" / "results")
    parser.add_argument("--fit-start", type=float, default=10.0)
    parser.add_argument("--fit-end", type=float, default=15.0)
    parser.add_argument("--slices", nargs="+", type=float, default=[10, 11, 12, 13, 14, 15])
    parser.add_argument("--stress-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    args = parser.parse_args()
    try:
        plot_rms_reynolds(args.data, args.output, args.fit_start, args.fit_end, args.slices, args.stress_sign)
    except Exception as exc:
        print(f"plot_rms_reynolds.py: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

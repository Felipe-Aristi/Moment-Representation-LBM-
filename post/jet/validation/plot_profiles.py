#!/usr/bin/env python3
"""MR-LBM round-jet mean profiles using the original jet solver's plot style."""
from pathlib import Path
import argparse
import sys

import numpy as np

# Reuse precisely the same prettyPlot.py and exportAndCrop.py as the old jet solver.
POST_DIR = Path(__file__).resolve().parents[2]
if str(POST_DIR) not in sys.path:
    sys.path.insert(0, str(POST_DIR))

from prettyPlot import pretty_plot
from exportAndCrop import export_and_crop
from post.channelflow.validation.statistics_io import DEFAULT_DATA_DIR, fit_virtual_origin, load_radial_profiles

red = "#E03A3E"
blue = "#009DDC"
green = "#61BB46"
purple = "#963D97"
yellow = "#FDB827"
orange = "#F5821F"
black = "#000000"
colors = [red, blue, green, purple, orange, yellow]
markers = ["o", "s", "^", "D", "v", "<", ">"]

# Original digitization used in the previous jet repository.
HUSSEIN_X = np.array([
    0, 0.010599057494345044, 0.019815647572123837, 0.030875562697158625,
    0.04009215277493742, 0.04838709790833872, 0.056221185411050346,
    0.06405527291376191, 0.07281104051935197, 0.08479260058876421,
    0.09400919066654304, 0.10737325331102249, 0.11658984338880131,
    0.12949308356109202, 0.14147464363050433, 0.15345620369991658,
    0.16866359139165193, 0.18248847650832015, 0.19723500656936588,
    0.21290321673329005, 0.22672810184995826, 0.2447004595333271,
])
HUSSEIN_U = np.array([
    0.9986970883174464, 0.988273596045439, 0.9622149647712097,
    0.9205211944947582, 0.8657980588782979, 0.8136807963298389,
    0.7563517876453765, 0.7042345250969178, 0.6469055164124551,
    0.5635178764537632, 0.514006536676201, 0.4332247200826162,
    0.38371338030505414, 0.32899019498569865, 0.2690553629361291,
    0.2247556698887811, 0.16742676061010808, 0.11791512261517773,
    0.08143319788051967, 0.05276854442960405, 0.03452768146806465,
    0.024104189196056974,
])

def first_half_radius(profile, radius, center_value):
    if not np.isfinite(center_value) or center_value <= 0.0:
        return np.nan
    ratio = profile / center_value
    finite = np.isfinite(ratio)
    below = np.where(finite & (ratio <= 0.5))[0]
    below = below[below > 0]
    if below.size == 0:
        return np.nan
    i = int(below[0])
    r1, r2 = radius[i - 1], radius[i]
    u1, u2 = ratio[i - 1], ratio[i]
    if not np.isfinite(u1) or not np.isfinite(u2) or np.isclose(u1, u2):
        return np.nan
    return r1 + (0.5 - u1) * (r2 - r1) / (u2 - u1)


def hussein_halfwidth_coordinate():
    """Hussein half-width in the original coordinate r/(y-y0)."""
    crossing = np.flatnonzero(HUSSEIN_U <= 0.5)
    if not crossing.size or crossing[0] == 0:
        raise ValueError("Hussein data do not bracket the half-centerline velocity")
    i = int(crossing[0])
    x1, x2 = HUSSEIN_X[i - 1], HUSSEIN_X[i]
    u1, u2 = HUSSEIN_U[i - 1], HUSSEIN_U[i]
    if u1 == u2:
        raise ValueError("Cannot interpolate Hussein half-width")
    return x1 + (0.5 - u1) * (x2 - x1) / (u2 - u1)


def _axis(xlim, ylim, xlabel, ylabel, xfmt=2, yfmt=1, nxticks=7, nyticks=6):
    fig, ax, _ = pretty_plot(
        xLim=xlim,
        yLim=ylim,
        cLim=(-1, 1),
        plotAspectRatio=(1, 1, 1),
        xLabel=xlabel,
        yLabel=ylabel,
        yScientificNotation=False,
        xTickFormat=xfmt,
        yTickFormat=yfmt,
        nxTicks=nxticks,
        nyTicks=nyticks,
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
    return fig, ax


def plot_mean_profiles(data_dir, output_dir, fit_start=10.0, fit_end=15.0,
                       slices=(10, 11, 12, 13, 14, 15)):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = load_radial_profiles(data_dir)
    uy, count = profile.uy, profile.count
    diameter = float(profile.metadata["D"])
    u_jet = float(profile.metadata["jet_velocity"])
    if diameter <= 0.0 or u_jet <= 0.0:
        raise ValueError("Jet diameter and inlet velocity must be positive")

    ny, nr = uy.shape
    y = np.arange(ny, dtype=np.float64)
    radius = np.arange(nr, dtype=np.float64)
    uc = uy[:, 0]
    valid_y = np.isfinite(uc) & (uc > 0.0) & (count[:, 0] > 0)
    if np.count_nonzero(valid_y) < 2:
        raise ValueError("Not enough valid centerline samples")

    spreading_rate, y0, x_fit, y_fit = fit_virtual_origin(
        y, uc, diameter, u_jet, fit_start, fit_end
    )
    slope = 1.0 / spreading_rate
    intercept = -y0 / (spreading_rate * diameter)
    print(f"Using mean profile folder: {data_dir}")
    print(f"B = {spreading_rate}")
    print(f"y0 = {y0}")

    inv_centerline = np.full_like(uc, np.nan)
    np.divide(u_jet, uc, out=inv_centerline, where=valid_y)
    inv_valid = inv_centerline[valid_y]

    fig1, ax1 = _axis(
        (0, ny / diameter), (0, np.ceil(np.nanmax(inv_valid))),
        r"$y/D$", r"$U_{jet}/U_{y,c}$", xfmt=5, yfmt=2, nxticks=9, nyticks=9
    )
    ax1.yaxis.set_label_coords(-0.125, 0.5)
    ax1.plot(y[valid_y] / diameter, inv_centerline[valid_y], ls="-", lw=3, color=black)
    export_and_crop(fig1, output_dir / "uy_centerline.png")

    fig2, ax2 = _axis(
        (fit_start, fit_end),
        (np.nanmin(y_fit) * 0.95, np.nanmax(y_fit) * 1.05),
        r"$y/D$", r"$U_{jet}/U_{y,c}$", xfmt=5, yfmt=2, nxticks=6, nyticks=9
    )
    ax2.yaxis.set_label_coords(-0.125, 0.5)
    ax2.plot(x_fit, y_fit, marker="o", mfc="none", ms=2, ls="none", lw=3, color=black, label="LBM")
    ax2.plot(x_fit, slope * x_fit + intercept, ls="--", lw=3, color=red,
             label=r"$(y-y_0)/(BD)$")
    ax2.legend(fontsize=14)
    export_and_crop(fig2, output_dir / "uy_centerline_fit.png")

    fig3, ax3 = _axis((0, 3), (0, 1.1), r"$r/D$",
                       r"$\langle u_y\rangle/U_{jet}$", xfmt=1)
    fig4, ax4 = _axis((0, 0.3), (0, 1.05), r"$r/(y-y_0)$",
                       r"$\langle u_y\rangle/U_{y,c}$", xfmt=2)
    fig5, ax5 = _axis((0, 3), (0, 1.05), r"$r/r_{1/2}$",
                       r"$\langle u_y\rangle/U_{y,c}$", xfmt=2)

    for ax in (ax3, ax4, ax5):
        ax.yaxis.set_label_coords(-0.125, 0.585)

    for j, slice_over_d in enumerate(slices):
        y_idx = int(round(slice_over_d * diameter))
        if y_idx <= 0 or y_idx >= ny or not np.isfinite(uc[y_idx]) or uc[y_idx] <= 0.0:
            print(f"Skipping y/D={slice_over_d}: outside domain or invalid centerline velocity.")
            continue
        values = uy[y_idx, :]
        valid_r = np.isfinite(values) & (count[y_idx, :] > 0)
        if not np.any(valid_r):
            print(f"Skipping y/D={slice_over_d}: no sampled radial bins.")
            continue
        color = colors[j % len(colors)]
        marker = markers[j % len(markers)]
        label = rf"$y={slice_over_d:g}D$"
        r_half = first_half_radius(values, radius, uc[y_idx])
        print(f"Slice {slice_over_d:g}D: r_half = {r_half:.4f}")

        ax3.plot(radius[valid_r] / diameter, values[valid_r] / u_jet,
                 lw=3, ls="none", marker=marker, ms=3, color=color, label=label)

        denom = y_idx - y0
        if denom > 0.0:
            # Hussein data are F(eta)=U/Uc, eta=r/(y-y0).
            # Convert to this station's r/D and U/Ujet using the LBM
            # virtual origin and centerline velocity. These are rescaled
            # experimental similarity profiles, not dimensional measurements.
            ax3.plot(HUSSEIN_X * denom / diameter,
                     HUSSEIN_U * uc[y_idx] / u_jet,
                     ls="--", lw=1.6, color=color, alpha=0.85,
                     label="Hussein et al. (1994), rescaled" if j == 0 else "_nolegend_")
            ax4.plot(radius[valid_r] / denom, values[valid_r] / uc[y_idx],
                     lw=3, ls="none", marker=marker, ms=3, color=color, label=label)

        if np.isfinite(r_half) and r_half > 0.0:
            ax5.plot(radius[valid_r] / r_half, values[valid_r] / uc[y_idx],
                     lw=3, ls="none", marker=marker, ms=3, color=color, label=label)

    ax4.plot(HUSSEIN_X, HUSSEIN_U, ls="none", marker="x", ms=4,
             color=black, label="Hussein et al. (1994)")

    # The experimental half-width corresponds to U/Uc = 0.5.
    # Scaling by it yields r/r_half, the horizontal coordinate of fig5.
    hussein_half = hussein_halfwidth_coordinate()
    ax5.plot(HUSSEIN_X / hussein_half, HUSSEIN_U, ls="none",
             marker="x", ms=4, color=black, label="Hussein et al. (1994)")
    print(f"Hussein half-width similarity coordinate: {hussein_half:.6f}")

    ax3.legend(fontsize=10)
    ax4.legend(fontsize=10)
    ax5.legend(fontsize=10)
    export_and_crop(fig3, output_dir / "uy_radial.png")
    export_and_crop(fig4, output_dir / "uy_selfsimilar.png")
    export_and_crop(fig5, output_dir / "uy_rhalf.png")


def main():
    parser = argparse.ArgumentParser(description="MR-LBM jet mean-flow profiles, original plot style")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "plots" / "results")
    parser.add_argument("--fit-start", type=float, default=10.0)
    parser.add_argument("--fit-end", type=float, default=15.0)
    parser.add_argument("--slices", nargs="+", type=float, default=[10, 11, 12, 13, 14, 15])
    args = parser.parse_args()
    try:
        plot_mean_profiles(args.data, args.output, args.fit_start, args.fit_end, args.slices)
    except Exception as exc:
        print(f"plot_profiles.py: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Mean-velocity validation for MR-LBM turbulent channel flow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib.ticker import (
    LogFormatterMathtext,
    LogLocator,
    NullFormatter,
)

# ============================================================
# Directories
# ============================================================

VALIDATION_DIR = Path(__file__).resolve().parent
POST_DIR = VALIDATION_DIR.parents[1]

if str(POST_DIR) not in sys.path:
    sys.path.insert(0, str(POST_DIR))

from prettyPlot import pretty_plot
from exportAndCrop import export_and_crop
from statistics_io import (
    load_channel_statistics,
    compute_channel_profiles,
)

REFERENCE_DIR = VALIDATION_DIR / "reference"

DNS_DEFAULT = REFERENCE_DIR / "chan180.means"

DNS_URL = (
    "https://turbulence.oden.utexas.edu/data/MKM/"
    "chan180/profiles/chan180.means"
)

OUTPUT_DEFAULT = VALIDATION_DIR / "plots" / "results"

# ============================================================
# Colors
# ============================================================

RED = "#E03A3E"
BLUE = "#009DDC"
GREEN = "#61BB46"
BLACK = "#000000"

# ============================================================
# Constants
# ============================================================

KAPPA = 0.41
B = 5.0

# visual ranges for the guide lines
VISCOUS_START = 0.5
VISCOUS_END = 12.0

LOG_START = 1.0

# ============================================================
# Read DNS file
# ============================================================

def read_dns_means(path: Path) -> tuple[np.ndarray, np.ndarray]:

    raw = np.loadtxt(
        path,
        comments="#",
        dtype=np.float64,
        ndmin=2,
    )

    if raw.shape[0] < 10 or raw.shape[1] < 3:
        raise ValueError(f"Invalid MKM DNS mean profile: {path}")

    y_plus = raw[:, 1]
    u_plus = raw[:, 2]

    if not np.all(np.isfinite(y_plus)) or not np.all(np.isfinite(u_plus)):
        raise ValueError("DNS file contains NaN or infinity")

    if np.any(y_plus < 0.0) or np.any(np.diff(y_plus) <= 0.0):
        raise ValueError("DNS y+ values are invalid")

    return y_plus, u_plus

# ============================================================
# Download DNS if needed
# ============================================================

def get_dns_data(path: Path, download: bool) -> tuple[np.ndarray, np.ndarray] | None:

    if not path.is_file() and download:

        print(f"Downloading MKM DNS reference: {DNS_URL}")

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")

        try:
            with urlopen(DNS_URL, timeout=15) as response:
                temporary.write_bytes(response.read())

            read_dns_means(temporary)
            temporary.replace(path)

        except (OSError, ValueError) as error:

            temporary.unlink(missing_ok=True)

            print(
                f"Warning: DNS download failed: {error}",
                file=sys.stderr,
            )

    if not path.is_file():

        print(
            "DNS data unavailable. Plotting MR-LBM only.",
            file=sys.stderr,
        )

        return None

    return read_dns_means(path)

# ============================================================
# Guide laws
# ============================================================

def viscous_sublayer(y_plus: np.ndarray) -> np.ndarray:
    return y_plus

def logarithmic_law(y_plus: np.ndarray) -> np.ndarray:
    return np.log(y_plus) / KAPPA + B

# ============================================================
# Plot
# ============================================================

def plot_mean_velocity(
    data_dir: Path | None = None,
    output_dir: Path = OUTPUT_DEFAULT,
    dns_file: Path = DNS_DEFAULT,
    download_dns: bool = True,
) -> Path:

    # ========================================================
    # Load MR-LBM data
    # ========================================================

    data = load_channel_statistics(data_dir)

    profiles = compute_channel_profiles(
        data,
        fold_walls=True,
    )

    y_plus = np.asarray(profiles.y_plus, dtype=np.float64)
    u_plus = np.asarray(profiles.ux_plus, dtype=np.float64)

    re_tau = float(data.metadata["ReTau"])

    valid = (
        np.isfinite(y_plus)
        & np.isfinite(u_plus)
        & (y_plus > 0.0)
    )

    if np.count_nonzero(valid) < 2:
        raise ValueError("Not enough valid MR-LBM points")

    # ========================================================
    # Load DNS
    # ========================================================

    dns = get_dns_data(Path(dns_file).expanduser(), download_dns)

    # ========================================================
    # Axis limits
    # ========================================================

    x_max = max(200.0, 1.05 * re_tau)
    u_max = max(18.0, float(np.nanmax(u_plus[valid])) * 1.08)

    if dns is not None:
        x_max = max(x_max, float(np.nanmax(dns[0])) * 1.05)
        u_max = max(u_max, float(np.nanmax(dns[1])) * 1.08)

    u_max = float(np.ceil(u_max / 2.0) * 2.0)

    # ========================================================
    # Figure
    # ========================================================

    fig, ax, _ = pretty_plot(
        xLim=(0.1, x_max),
        yLim=(0.0, u_max),

        xLabel=r"$y^+$",
        yLabel=r"$U^+$",

        xTickFormat=5,
        yTickFormat=0,

        nxTicks=5,
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

    # ========================================================
    # Logarithmic x-axis
    # ========================================================

    ax.set_xscale("log")

    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.set_xlim(0.1, x_max)
    ax.set_ylim(0.0, u_max)

    ax.yaxis.set_label_coords(-0.12, 0.5)

    # ========================================================
    # DNS
    # ========================================================

    if dns is not None:

        y_dns, u_dns = dns

        valid_dns = (
            np.isfinite(y_dns)
            & np.isfinite(u_dns)
            & (y_dns > 0.0)
            & (y_dns <= x_max)
        )

        ax.plot(
            y_dns[valid_dns],
            u_dns[valid_dns],
            linestyle="none",
            marker="+",
            markersize=5,
            markeredgewidth=1.0,
            color=BLACK,
            label=r"DNS (MKM 1999, $Re_\tau=178.1$)",
            zorder=3,
        )

    # ========================================================
    # Blue dashed line: U+ = y+
    #
    # plot only in the near-wall visual range
    # ========================================================

    viscous_end = min(VISCOUS_END, x_max, u_max)

    viscous_x = np.geomspace(
        VISCOUS_START,
        viscous_end,
        300,
    )
    viscous_y = viscous_sublayer(viscous_x)

    ax.plot(
        viscous_x,
        viscous_y,
        linestyle="--",
        linewidth=1.8,
        color=BLUE,
        label=r"$U^+=y^+$",
        zorder=1,
    )

    # ========================================================
    # Green dashed line: U+ = ln(y+)/0.41 + 5
    #
    # extend it through the whole visible range,
    # then clip to the y-limits of the figure
    # ========================================================

    log_x = np.geomspace(
        LOG_START,
        x_max,
        600,
    )
    log_y = logarithmic_law(log_x)

    mask_log = (
        np.isfinite(log_y)
        & (log_y >= 0.0)
        & (log_y <= u_max)
    )

    ax.plot(
        log_x[mask_log],
        log_y[mask_log],
        linestyle="--",
        linewidth=1.8,
        color=GREEN,
        label=r"$U^+=\ln(y^+)/0.41+5$",
        zorder=1,
    )

    # ========================================================
    # MR-LBM
    # ========================================================

    ax.plot(
        y_plus[valid],
        u_plus[valid],
        linestyle="none",
        marker="o",
        markersize=3.6,
        markerfacecolor="none",
        markeredgewidth=1.0,
        color=RED,
        label=rf"MR-LBM ($Re_\tau={re_tau:g}$)",
        zorder=4,
    )

    # ========================================================
    # Legend
    # ========================================================

    ax.legend(
        loc="lower right",
        fontsize=10,
        frameon=False,
    )

    # ========================================================
    # Save
    # ========================================================

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "mean_velocity.png"

    export_and_crop(fig, output_path)

    print(f"Channel data: {data.directory}")
    print(f"Samples: {data.metadata['sample_count']}")
    print(f"Re_tau (MR-LBM): {re_tau:g}")
    print(f"First fluid y+: {float(np.min(y_plus[valid])):.6g}")
    print(f"Saved: {output_path.resolve()}")

    return output_path

# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="MR-LBM channel mean velocity compared with MKM DNS"
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Channel statistics directory",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DEFAULT,
        help="Output directory",
    )

    parser.add_argument(
        "--dns",
        type=Path,
        default=DNS_DEFAULT,
        help="Local MKM chan180.means file",
    )

    parser.add_argument(
        "--no-download-dns",
        action="store_true",
        help="Do not download missing DNS data",
    )

    args = parser.parse_args()

    try:
        plot_mean_velocity(
            data_dir=args.data,
            output_dir=args.output,
            dns_file=args.dns,
            download_dns=not args.no_download_dns,
        )

    except (OSError, ValueError, KeyError) as error:
        parser.exit(
            1,
            f"plot_mean_velocity.py: {error}\n",
        )

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""MR-LBM channel: RMS velocities and Reynolds shear stress vs MKM DNS.

Montessori et al. (2024), Figure 5. Channel coordinates in this solver:
x = streamwise, y = wall-normal, z = spanwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

VALIDATION_DIR = Path(__file__).resolve().parent
POST_DIR = VALIDATION_DIR.parents[1]
if str(POST_DIR) not in sys.path:
    sys.path.insert(0, str(POST_DIR))

from prettyPlot import pretty_plot
from exportAndCrop import export_and_crop
from statistics_io import load_channel_statistics, compute_channel_profiles

# ============================================================
# Directories
# ============================================================

REFERENCE_DIR = VALIDATION_DIR / "reference"
DNS_DEFAULT = REFERENCE_DIR / "chan180.reystress"

DNS_URL = (
    "https://turbulence.oden.utexas.edu/data/MKM/"
    "chan180/profiles/chan180.reystress"
)

OUTPUT_DEFAULT = VALIDATION_DIR / "plots" / "results"

# ============================================================
# Original jet plotting colors
# ============================================================

RED = "#E03A3E"
BLUE = "#009DDC"
GREEN = "#61BB46"
PURPLE = "#963D97"

# ============================================================
# Statistics to compare
#
# Label, channel profile, DNS profile, color, marker
# ============================================================

QUANTITIES = (
    (
        r"$u_{x,\mathrm{rms}}^+$",
        "ux_rms_plus", "ux_rms_plus", RED, "o"
    ),
    (
        r"$u_{y,\mathrm{rms}}^+$",
        "uy_rms_plus", "uy_rms_plus", BLUE, "s"
    ),
    (
        r"$u_{z,\mathrm{rms}}^+$",
        "uz_rms_plus", "uz_rms_plus", GREEN, "^"
    ),
    (
        r"$-\langle u_x'u_y'\rangle/u_\tau^2$",
        "minus_reynolds_uxuy_plus",
        "minus_reynolds_uxuy_plus",
        PURPLE, "D"
    ),
)

# ============================================================
# Read Moser-Kim-Mansour DNS
#
# Columns:
# 0 y/h
# 1 y+
# 2 Ruu
# 3 Rvv
# 4 Rww
# 5 Ruv
# 6 Ruw
# 7 Rvw
#
# DNS stresses are normalized by u_tau^2.
# Diagonal stresses must be square-rooted for RMS.
# ============================================================

def read_dns_reynolds_stresses(path: Path) -> dict[str, np.ndarray]:

    raw = np.loadtxt(
        path,
        comments="#",
        dtype=np.float64,
        ndmin=2,
    )

    if raw.shape[0] < 10 or raw.shape[1] != 8:
        raise ValueError(
            f"Invalid MKM Reynolds-stress file: {path}"
        )

    if not np.isfinite(raw).all():
        raise ValueError("DNS data contain NaN or infinity")

    if np.any(raw[:, 1] < 0) or np.any(np.diff(raw[:, 1]) <= 0):
        raise ValueError("DNS y+ coordinates are invalid")

    variances = raw[:, 2:5]

    if np.any(variances < -1e-12):
        raise ValueError("Negative DNS diagonal Reynolds stress")

    return {
        "y_plus": raw[:, 1],

        "ux_rms_plus": np.sqrt(np.maximum(raw[:, 2], 0.0)),
        "uy_rms_plus": np.sqrt(np.maximum(raw[:, 3], 0.0)),
        "uz_rms_plus": np.sqrt(np.maximum(raw[:, 4], 0.0)),

        "minus_reynolds_uxuy_plus": -raw[:, 5],
    }


# ============================================================
# Load or download DNS reference
# ============================================================

def load_dns(
    path: Path,
    allow_download: bool,
) -> dict[str, np.ndarray] | None:

    if not path.is_file() and allow_download:

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")

        try:
            print(f"Downloading MKM DNS: {DNS_URL}")

            with urlopen(DNS_URL, timeout=15) as response:
                temporary.write_bytes(response.read())

            read_dns_reynolds_stresses(temporary)
            temporary.replace(path)

        except (OSError, ValueError) as exc:
            temporary.unlink(missing_ok=True)
            print(
                f"Warning: DNS download failed: {exc}",
                file=sys.stderr,
            )

    if not path.is_file():
        print(
            "DNS unavailable. Plotting MR-LBM only.",
            file=sys.stderr,
        )
        return None

    return read_dns_reynolds_stresses(path)


# ============================================================
# Reynolds-stress validation
# ============================================================

def plot_reynolds_stresses(
    data_dir: Path | None = None,
    output_dir: Path = OUTPUT_DEFAULT,
    dns_path: Path = DNS_DEFAULT,
    download_dns: bool = True,
) -> Path:

    # ========================================================
    # MR-LBM statistics
    # ========================================================

    data = load_channel_statistics(data_dir)

    profiles = compute_channel_profiles(
        data,
        fold_walls=True,
    )

    y_lbm = np.asarray(
        profiles.y_plus,
        dtype=np.float64,
    )

    valid_x = np.isfinite(y_lbm) & (y_lbm > 0)

    if np.count_nonzero(valid_x) < 2:
        raise ValueError(
            "Not enough wall-normal points for plotting"
        )

    # ========================================================
    # DNS
    # ========================================================

    dns = load_dns(
        Path(dns_path).expanduser(),
        download_dns,
    )

    # ========================================================
    # Axis limits
    # ========================================================

    x_max = max(
        200.0,
        1.05 * float(np.max(y_lbm[valid_x])),
    )

    upper = 3.2
    lower = -0.15

    for _, lbm_attr, dns_key, _, _ in QUANTITIES:

        values = np.asarray(
            getattr(profiles, lbm_attr),
            dtype=np.float64,
        )

        finite = np.isfinite(values) & valid_x

        if not np.any(finite):
            raise ValueError(
                f"No finite data for {lbm_attr}"
            )

        upper = max(
            upper,
            1.10 * float(np.max(values[finite])),
        )

        lower = min(
            lower,
            1.10 * float(np.min(values[finite])),
        )

        if dns is not None:
            upper = max(
                upper,
                1.10 * float(np.max(dns[dns_key])),
            )

            lower = min(
                lower,
                1.10 * float(np.min(dns[dns_key])),
            )

    upper = float(np.ceil(upper * 2) / 2)
    lower = float(np.floor(lower * 2) / 2)

    # ========================================================
    # Figure - same aesthetics as turbulent jet
    # ========================================================

    fig, ax, _ = pretty_plot(
        xLim=(0.1, x_max),
        yLim=(lower, upper),

        xLabel=r"$y^+$",
        yLabel=(
            r"$u_{i,\mathrm{rms}}^+,\ "
            r"-\langle u_x'u_y'\rangle/u_\tau^2$"
        ),

        xTickFormat=5,
        yTickFormat=1,

        nxTicks=5,
        nyTicks=8,

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
    # Logarithmic y+ axis
    # ========================================================

    ax.set_xscale("log")

    ax.xaxis.set_major_locator(
        LogLocator(base=10)
    )

    ax.xaxis.set_major_formatter(
        LogFormatterMathtext(base=10)
    )

    ax.xaxis.set_minor_locator(
        LogLocator(
            base=10,
            subs=np.arange(2, 10),
        )
    )

    ax.xaxis.set_minor_formatter(
        NullFormatter()
    )

    ax.set_xlim(0.1, x_max)
    ax.set_ylim(lower, upper)

    ax.yaxis.set_label_coords(-0.15, 0.5)


    # ========================================================
    # Plot all four statistical quantities
    #
    # DNS: plus markers
    # MR-LBM: open markers
    #
    # Matching colors identify the same quantity.
    # ========================================================

    for label, lbm_attr, dns_key, color, marker in QUANTITIES:

        values = np.asarray(
            getattr(profiles, lbm_attr),
            dtype=np.float64,
        )

        valid = valid_x & np.isfinite(values)

        # ====================================================
        # DNS
        # ====================================================

        if dns is not None:

            mask = (
                (dns["y_plus"] > 0)
                & (dns["y_plus"] <= x_max)
            )

            ax.plot(
                dns["y_plus"][mask],
                dns[dns_key][mask],

                linestyle="none",
                marker="+",

                markersize=5,
                markeredgewidth=1.0,

                color=color,

                label=f"DNS {label}",

                zorder=2,
            )

        # ====================================================
        # MR-LBM
        # ====================================================

        x = y_lbm[valid]
        v = values[valid]

        # Avoid excessive marker overlap.
        marker_step = max(
            1,
            int(np.ceil(x.size / 90)),
        )

        ax.plot(
            x,
            v,

            linestyle="none",
            marker=marker,

            markevery=marker_step,

            markersize=4.5,
            markerfacecolor="none",
            markeredgewidth=1.0,

            color=color,

            label=f"MR-LBM {label}",

            zorder=3,
        )

    # ========================================================
    # Legend
    # ========================================================

    ax.legend(
        loc="upper left",
        fontsize=9,
        frameon=False,
        ncol=2,
        columnspacing=1.3,
        handletextpad=0.4,
        borderaxespad=0.5,
    )

    # ========================================================
    # Export
    # ========================================================

    output_dir = Path(output_dir).expanduser()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "reynolds_stresses.png"

    export_and_crop(
        fig,
        output_path,
    )

    # ========================================================
    # Terminal output
    # ========================================================

    print(f"Channel data: {data.directory}")
    print(f"Samples: {data.metadata['sample_count']}")
    print(f"Re_tau (MR-LBM): {data.metadata['ReTau']}")

    print(
        f"DNS: {dns_path if dns is not None else 'not available'}"
    )

    print(f"Saved: {output_path.resolve()}")

    return output_path


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "MR-LBM channel RMS velocities and Reynolds "
            "shear stress compared with MKM DNS"
        )
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
        help="Output figure directory",
    )

    parser.add_argument(
        "--dns",
        type=Path,
        default=DNS_DEFAULT,
        help="Local MKM chan180.reystress file",
    )

    parser.add_argument(
        "--no-download-dns",
        action="store_true",
        help="Do not download missing DNS data",
    )

    args = parser.parse_args()

    try:
        plot_reynolds_stresses(
            args.data,
            args.output,
            args.dns,
            download_dns=not args.no_download_dns,
        )

    except (OSError, ValueError, KeyError) as exc:
        parser.exit(
            1,
            f"plot_reynolds_stresses.py: {exc}\n",
        )


if __name__ == "__main__":
    main()

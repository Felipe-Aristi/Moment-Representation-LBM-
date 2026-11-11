
#!/usr/bin/env python3
"""MR-LBM channel-flow skewness validation.

Reproduces the comparison in Montessori et al. (2024), Fig. 6.

Coordinates:
    x = streamwise
    y = wall-normal
    z = spanwise

DNS reference:
    Moser, Kim & Mansour (1999), Re_tau = 178.12.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

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

# ============================================================
# Directories
# ============================================================

REFERENCE_DIR = VALIDATION_DIR / "reference"

DNS_DEFAULT = REFERENCE_DIR / "chan180.skew"

DNS_URL = (
    "https://turbulence.oden.utexas.edu/data/MKM/"
    "chan180/profiles/chan180.skew"
)

OUTPUT_DEFAULT = VALIDATION_DIR / "plots" / "results"

# ============================================================
# Original jet plotting colors
# ============================================================

RED = "#E03A3E"
BLUE = "#009DDC"
GREEN = "#61BB46"

# ============================================================
# Quantities
#
# Label, LBM attribute, DNS key, color, LBM marker
# ============================================================

QUANTITIES = (
    (
        r"$S_x$",
        "skewness_ux",
        "skewness_ux",
        RED,
        "o",
    ),
    (
        r"$S_y$",
        "skewness_uy",
        "skewness_uy",
        BLUE,
        "s",
    ),
    (
        r"$S_z$",
        "skewness_uz",
        "skewness_uz",
        GREEN,
        "^",
    ),
)

# ============================================================
# Read Moser-Kim-Mansour skewness reference
#
# chan180.skew columns:
#
# 0: y/h
# 1: y+
# 2: S(u) -> streamwise
# 3: S(v) -> wall-normal
# 4: S(w) -> spanwise
# 5: S(p) -> pressure (not used)
# ============================================================

def read_dns_skewness(
    path: Path,
) -> dict[str, np.ndarray]:

    raw = np.loadtxt(
        path,
        comments="#",
        dtype=np.float64,
        ndmin=2,
    )

    if raw.shape[0] < 10 or raw.shape[1] != 6:
        raise ValueError(
            f"Invalid MKM DNS skewness file: {path}"
        )

    if not np.all(np.isfinite(raw)):
        raise ValueError(
            "DNS skewness contains NaN or infinity"
        )

    y_plus = raw[:, 1]

    if (
        np.any(y_plus < 0)
        or np.any(np.diff(y_plus) <= 0)
    ):
        raise ValueError(
            "DNS wall coordinates are invalid"
        )

    return {
        "y_plus": y_plus,
        "skewness_ux": raw[:, 2],
        "skewness_uy": raw[:, 3],
        "skewness_uz": raw[:, 4],
    }


# ============================================================
# Load or download DNS reference
# ============================================================

def load_dns(
    path: Path,
    allow_download: bool,
) -> dict[str, np.ndarray] | None:

    if not path.is_file() and allow_download:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = path.with_name(
            path.name + ".tmp"
        )

        try:
            print(f"Downloading DNS skewness: {DNS_URL}")

            with urlopen(DNS_URL, timeout=15) as response:
                temporary.write_bytes(
                    response.read()
                )

            # Check the data before keeping the download.
            read_dns_skewness(temporary)

            temporary.replace(path)

        except (OSError, ValueError) as exc:

            temporary.unlink(missing_ok=True)

            print(
                f"Warning: DNS download failed: {exc}",
                file=sys.stderr,
            )

    if not path.is_file():

        print(
            "DNS skewness unavailable. "
            "Plotting MR-LBM data only.",
            file=sys.stderr,
        )

        return None

    return read_dns_skewness(path)


# ============================================================
# Plot velocity skewness
# ============================================================

def plot_skewness(
    data_dir: Path | None = None,
    output_dir: Path = OUTPUT_DEFAULT,
    dns_path: Path = DNS_DEFAULT,
    download_dns: bool = True,
    max_y_plus: float = 80.0,
    y_min: float = -2.0,
    y_max: float = 2.0,
) -> Path:

    if max_y_plus <= 0:
        raise ValueError(
            "Maximum wall coordinate must be positive"
        )

    if y_max <= y_min:
        raise ValueError(
            "Invalid skewness axis limits"
        )

    # ========================================================
    # Load MR-LBM statistics
    #
    # Fold upper and lower walls.
    # The statistics reader handles the sign reversal
    # of wall-normal odd moments.
    # ========================================================

    data = load_channel_statistics(
        data_dir
    )

    profiles = compute_channel_profiles(
        data,
        fold_walls=True,
    )

    y_lbm = np.asarray(
        profiles.y_plus,
        dtype=np.float64,
    )

    # ========================================================
    # Load DNS reference
    # ========================================================

    dns = load_dns(
        Path(dns_path).expanduser(),
        download_dns,
    )

    # ========================================================
    # Figure aesthetics
    # ========================================================

    fig, ax, _ = pretty_plot(
        xLim=(0.0, max_y_plus),
        yLim=(y_min, y_max),

        xLabel=r"$y^+$",
        yLabel=r"$S_i$",

        xTickFormat=0,
        yTickFormat=1,

        nxTicks=9,
        nyTicks=9,

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

    ax.set_xlim(
        0.0,
        max_y_plus,
    )

    ax.set_ylim(
        y_min,
        y_max,
    )

    ax.yaxis.set_label_coords(
        -0.12,
        0.5,
    )


    # ========================================================
    # Plot all three skewness components
    #
    # DNS: plus markers
    # MR-LBM: open markers
    #
    # Same color identifies the same component.
    # ========================================================

    for label, lbm_attr, dns_key, color, marker in QUANTITIES:

        values = np.asarray(
            getattr(profiles, lbm_attr),
            dtype=np.float64,
        )

        valid = (
            np.isfinite(y_lbm)
            & np.isfinite(values)
            & (y_lbm > 0)
            & (y_lbm <= max_y_plus)
        )

        if np.count_nonzero(valid) < 2:
            raise ValueError(
                f"Not enough finite LBM data for {lbm_attr}"
            )

        x = y_lbm[valid]
        s = values[valid]

        # ====================================================
        # MR-LBM markers
        # ====================================================

        marker_step = max(
            1,
            int(np.ceil(x.size / 75)),
        )

        ax.plot(
            x,
            s,

            linestyle="none",
            marker=marker,

            markevery=marker_step,

            markersize=4.0,
            markerfacecolor="none",
            markeredgewidth=1.0,

            color=color,

            label=f"MR-LBM {label}",

            zorder=3,
        )

        # ====================================================
        # DNS reference
        # ====================================================

        if dns is not None:

            y_dns = dns["y_plus"]
            s_dns = dns[dns_key]

            valid_dns = (
                np.isfinite(y_dns)
                & np.isfinite(s_dns)
                & (y_dns >= 0)
                & (y_dns <= max_y_plus)
            )

            ax.plot(
                y_dns[valid_dns],
                s_dns[valid_dns],

                linestyle="none",
                marker="+",

                markersize=5,
                markeredgewidth=1.1,

                color=color,

                label=f"DNS {label}",

                zorder=2,
            )

        # ====================================================
        # Numerical diagnostics
        # ====================================================

        print(
            f"{lbm_attr}: "
            f"min={np.min(s):.6f}, "
            f"max={np.max(s):.6f}"
        )

        if np.any((s < y_min) | (s > y_max)):
            print(
                f"Warning: {lbm_attr} has values outside "
                f"the plot limits ({y_min}, {y_max}). "
                "Inspect FP32 raw-moment accuracy.",
                file=sys.stderr,
            )

    # ========================================================
    # Legend
    # ========================================================

    ax.legend(
        loc="upper right",
        fontsize=10,
        frameon=False,

        ncol=2,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    # ========================================================
    # Export
    # ========================================================

    output_dir = Path(output_dir).expanduser()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "skewness.png"

    export_and_crop(
        fig,
        output_path,
    )

    # ========================================================
    # Terminal output
    # ========================================================

    print(f"Channel data: {data.directory}")

    print(
        f"Samples: {data.metadata['sample_count']}"
    )

    print(
        f"Re_tau (MR-LBM): {data.metadata['ReTau']}"
    )

    print(
        f"DNS: {dns_path if dns is not None else 'not available'}"
    )

    print(
        f"Saved: {output_path.resolve()}"
    )

    return output_path


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "MR-LBM channel velocity skewness "
            "compared with MKM DNS"
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
        help="Output directory",
    )

    parser.add_argument(
        "--dns",
        type=Path,
        default=DNS_DEFAULT,
        help="Local MKM chan180.skew file",
    )

    parser.add_argument(
        "--no-download-dns",
        action="store_true",
        help="Do not download missing DNS data",
    )

    parser.add_argument(
        "--max-y-plus",
        type=float,
        default=80.0,
        help="Maximum wall coordinate (default: 80)",
    )

    parser.add_argument(
        "--y-min",
        type=float,
        default=-2.0,
        help="Minimum skewness axis value",
    )

    parser.add_argument(
        "--y-max",
        type=float,
        default=2.0,
        help="Maximum skewness axis value",
    )

    args = parser.parse_args()

    try:
        plot_skewness(
            data_dir=args.data,
            output_dir=args.output,
            dns_path=args.dns,

            download_dns=not args.no_download_dns,

            max_y_plus=args.max_y_plus,
            y_min=args.y_min,
            y_max=args.y_max,
        )

    except (OSError, ValueError, KeyError) as exc:

        parser.exit(
            1,
            f"plot_skewness.py: {exc}\n",
        )


if __name__ == "__main__":
    main()

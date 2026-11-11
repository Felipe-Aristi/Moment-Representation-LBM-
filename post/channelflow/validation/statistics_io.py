
#!/usr/bin/env python3
"""Read FP32 wall-normal statistics from the MR-LBM turbulent channel solver."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ============================================================
# Directories and moment definitions
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = BASE_DIR / "data"

MOMENT_NAMES = (
    "ux", "uy", "uz",
    "ux2", "uy2", "uz2",
    "uxuy",
    "ux3", "uy3", "uz3",
)

# Reflect upper-wall statistics into lower-wall coordinates.
WALL_REFLECTION_SIGNS = (1, -1, 1, 1, 1, 1, -1, 1, -1, 1)


# ============================================================
# Data structures
# ============================================================

@dataclass
class ChannelData:
    directory: Path
    metadata: dict[str, str]
    moments: dict[str, np.ndarray]


@dataclass
class ChannelProfiles:
    y: np.ndarray
    y_plus: np.ndarray

    ux_plus: np.ndarray
    uy_plus: np.ndarray
    uz_plus: np.ndarray

    ux_rms_plus: np.ndarray
    uy_rms_plus: np.ndarray
    uz_rms_plus: np.ndarray

    reynolds_uxuy_plus: np.ndarray
    minus_reynolds_uxuy_plus: np.ndarray

    skewness_ux: np.ndarray
    skewness_uy: np.ndarray
    skewness_uz: np.ndarray

    metadata: dict[str, str]
    folded: bool


# ============================================================
# Read metadata
# ============================================================

def read_metadata(directory: str | Path) -> dict[str, str]:

    path = Path(directory) / "channel_statistics_metadata.txt"

    if not path.is_file():
        raise FileNotFoundError(f"Missing channel metadata: {path}")

    metadata = {}

    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = line.split("#", 1)[0].strip()

        if not line:
            continue

        fields = line.split(maxsplit=1)

        if len(fields) != 2 or fields[0] in metadata:
            raise ValueError(
                f"Invalid or repeated metadata line {number}: {line}"
            )

        metadata[fields[0]] = fields[1]

    return metadata


# ============================================================
# Locate simulation data
# ============================================================

def find_channel_data(data: str | Path | None = None) -> Path:

    if data is not None:
        directory = Path(data).expanduser().resolve()

        if not directory.is_dir():
            raise FileNotFoundError(
                f"Channel data directory not found: {directory}"
            )

        return directory

    candidates = list(
        DEFAULT_DATA_ROOT.glob(
            "RETAU*_*/channel_statistics_metadata.txt"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            f"No channel statistics found under {DEFAULT_DATA_ROOT}. "
            "Run the CUDA channel solver until its first statistics "
            "checkpoint, or specify --data PATH."
        )

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    ).parent


# ============================================================
# Load FP32 raw moments
#
# Binary layout:
# moment * NY + y
# ============================================================

def load_channel_statistics(
    data: str | Path | None = None,
) -> ChannelData:

    directory = find_channel_data(data)
    metadata = read_metadata(directory)

    if metadata.get("moment_dtype") != "float32":
        raise ValueError("Expected FP32 channel moments")

    if metadata.get("moment_layout") != \
            "field_major_field_times_NY_plus_y":
        raise ValueError("Unexpected channel binary layout")

    if metadata.get("wall_normal") != "y":
        raise ValueError("Expected wall-normal direction y")

    if int(metadata.get("moment_fields", "-1")) != len(MOMENT_NAMES):
        raise ValueError("Expected ten statistical moments")

    if tuple(metadata.get("moment_names", "").split()) != MOMENT_NAMES:
        raise ValueError("Channel moment ordering does not match the reader")

    ny = int(metadata["NY"])
    sample_count = int(metadata["sample_count"])

    if ny < 3 or sample_count <= 0:
        raise ValueError("Invalid channel dimensions or sample count")

    path = directory / "channel_moments.bin"

    expected_bytes = len(MOMENT_NAMES) * ny * np.dtype("<f4").itemsize
    actual_bytes = path.stat().st_size

    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{path}: found {actual_bytes} bytes, "
            f"expected {expected_bytes}"
        )

    raw = np.fromfile(
        path,
        dtype="<f4",
    ).reshape(len(MOMENT_NAMES), ny)

    if not np.isfinite(raw).all():
        raise ValueError("Channel moments contain NaN or infinity")

    # Convert to FP64 only for post-processing calculations.
    moments = {
        name: np.asarray(raw[i], dtype=np.float64)
        for i, name in enumerate(MOMENT_NAMES)
    }

    # ========================================================
    # Verify wall-unit normalization
    # ========================================================

    half_height = float(metadata["halfHeight"])
    u_tau = float(metadata["uTau"])
    nu = float(metadata["nu"])
    re_tau = float(metadata["ReTau"])

    if not (
        half_height > 0
        and u_tau > 0
        and nu > 0
        and re_tau > 0
    ):
        raise ValueError("Invalid channel physical parameters")

    if not np.isclose(
        half_height,
        (ny - 1) / 2,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError("Inconsistent channel half-height")

    if not np.isclose(
        u_tau * half_height / nu,
        re_tau,
        rtol=1e-4,
        atol=1e-4,
    ):
        raise ValueError("Inconsistent friction Reynolds number")

    return ChannelData(
        directory=directory,
        metadata=metadata,
        moments=moments,
    )


# ============================================================
# RMS and skewness
# ============================================================

def _skewness(
    mean: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    variance = second - mean * mean

    if np.any(
        variance < -1e-5 * np.maximum(second, 1e-12)
    ):
        raise ValueError(
            "Substantially negative variance: "
            "check FP32 statistical accumulation"
        )

    variance = np.maximum(variance, 0.0)

    central_third = (
        third
        - 3.0 * mean * second
        + 2.0 * mean**3
    )

    skewness = np.full_like(variance, np.nan)

    # Skewness is undefined at zero variance.
    valid = variance > 0.0

    np.divide(
        central_third,
        variance**1.5,
        out=skewness,
        where=valid,
    )

    return np.sqrt(variance), skewness


# ============================================================
# Compute channel profiles
# ============================================================

def compute_channel_profiles(
    data: ChannelData,
    fold_walls: bool = True,
) -> ChannelProfiles:

    m = data.moments
    ny = int(data.metadata["NY"])

    # ========================================================
    # Fold upper and lower channel halves
    # ========================================================

    if fold_walls:
        n_half = (ny + 1) // 2

        moments = {
            name: 0.5 * (
                m[name][:n_half]
                + sign * m[name][::-1][:n_half]
            )
            for name, sign in zip(
                MOMENT_NAMES,
                WALL_REFLECTION_SIGNS,
            )
        }

        y = np.arange(n_half, dtype=np.float64)

    else:
        moments = m
        y = np.arange(ny, dtype=np.float64)

    # ========================================================
    # First-order moments
    # ========================================================

    ux = moments["ux"]
    uy = moments["uy"]
    uz = moments["uz"]

    # ========================================================
    # RMS and skewness
    # ========================================================

    ux_rms, skewness_ux = _skewness(
        ux,
        moments["ux2"],
        moments["ux3"],
    )

    uy_rms, skewness_uy = _skewness(
        uy,
        moments["uy2"],
        moments["uy3"],
    )

    uz_rms, skewness_uz = _skewness(
        uz,
        moments["uz2"],
        moments["uz3"],
    )

    # ========================================================
    # Reynolds shear stress
    # ========================================================

    reynolds_uxuy = moments["uxuy"] - ux * uy

    # ========================================================
    # Wall units
    # ========================================================

    u_tau = float(data.metadata["uTau"])
    nu = float(data.metadata["nu"])

    wall_distance = (
        y if fold_walls
        else np.minimum(y, ny - 1 - y)
    )

    y_plus = wall_distance * u_tau / nu

    return ChannelProfiles(
        y=y,
        y_plus=y_plus,

        ux_plus=ux / u_tau,
        uy_plus=uy / u_tau,
        uz_plus=uz / u_tau,

        ux_rms_plus=ux_rms / u_tau,
        uy_rms_plus=uy_rms / u_tau,
        uz_rms_plus=uz_rms / u_tau,

        reynolds_uxuy_plus=reynolds_uxuy / (u_tau * u_tau),
        minus_reynolds_uxuy_plus=-reynolds_uxuy / (u_tau * u_tau),

        skewness_ux=skewness_ux,
        skewness_uy=skewness_uy,
        skewness_uz=skewness_uz,

        metadata=data.metadata,
        folded=fold_walls,
    )


# ============================================================
# Command-line arguments
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Inspect MR-LBM turbulent channel statistics"
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Channel data directory (default: latest run)",
    )

    parser.add_argument(
        "--no-fold",
        action="store_true",
        help="Keep the upper and lower channel halves separate",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    data = load_channel_statistics(args.data)

    profiles = compute_channel_profiles(
        data,
        fold_walls=not args.no_fold,
    )

    print("Data directory:", data.directory)
    print("Samples:", data.metadata["sample_count"])

    print(
        "Steps:",
        data.metadata["first_step"],
        "to",
        data.metadata["last_step"],
    )

    print("Re_tau:", data.metadata["ReTau"])
    print("Wall-normal points:", profiles.y.size)

    print(
        "Wall folding:",
        "enabled" if profiles.folded else "disabled",
    )

    print("Maximum y+:", np.max(profiles.y_plus))
    print("Maximum mean U+:", np.nanmax(profiles.ux_plus))
    print("Maximum streamwise RMS+:", np.nanmax(profiles.ux_rms_plus))

    print(
        "Finite streamwise skewness points:",
        np.isfinite(profiles.skewness_ux).sum(),
    )

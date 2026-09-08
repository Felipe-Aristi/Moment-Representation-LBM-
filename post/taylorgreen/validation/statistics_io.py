#!/usr/bin/env python3
"""Read and validate Taylor-Green scalar statistics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np


VALIDATION_DIRECTORY: Final = Path(__file__).resolve().parent

DEFAULT_CASE_NAME: Final = "TG_RE_1600"

METADATA_FILENAME: Final = "statistics_metadata.txt"
KINETIC_ENERGY_FILENAME: Final = "kinetic_energy.bin"
ENSTROPHY_FILENAME: Final = "enstrophy.bin"

EXPECTED_INITIAL_KSTAR: Final = 0.125
EXPECTED_CONTINUUM_INITIAL_ZETASTAR: Final = 0.375

INITIAL_RTOL: Final = 1.0e-5


_REQUIRED_METADATA_KEYS: Final = frozenset(
    {
        "format_version",
        "kinetic_energy_file",
        "enstrophy_file",
        "Re",
        "NX",
        "NY",
        "NZ",
        "NXPhysical",
        "NYPhysical",
        "NZPhysical",
        "Nphysical",
        "tgv_U0",
        "tgv_kx",
        "tgv_ky",
        "tgv_kz",
        "tgv_L",
        "Steps",
        "statisticsStart",
        "statisticsEvery",
        "record_step_dtype",
        "record_value_dtype",
        "record_value_encoding",
        "record_byte_order",
        "record_layout",
        "record_step_offset_bytes",
        "record_value_offset_bytes",
        "record_step_bytes",
        "record_value_bytes",
        "record_bytes",
        "kinetic_energy_definition",
        "enstrophy_definition",
        "vorticity_derivative",
        "grid_spacing",
    }
)


class StatisticsFormatError(ValueError):
    """Raised when Taylor-Green statistics files are inconsistent."""


@dataclass(frozen=True)
class KineticEnergyHistory:
    """Taylor-Green kinetic-energy history."""

    step: np.ndarray
    SK: np.ndarray
    tStar: np.ndarray
    KStar: np.ndarray
    metadata: dict[str, str]
    case_directory: Path


@dataclass(frozen=True)
class EnstrophyHistory:
    """Taylor-Green enstrophy history."""

    step: np.ndarray
    Szeta: np.ndarray
    tStar: np.ndarray
    ZetaStar: np.ndarray
    metadata: dict[str, str]
    case_directory: Path


def _format_error(path: Path, message: str) -> StatisticsFormatError:
    return StatisticsFormatError(f"{path}: {message}")


# ============================================================
# Metadata reader
# ============================================================

def read_statistics_metadata(path: str | Path) -> dict[str, str]:
    metadata_path = Path(path)

    try:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise StatisticsFormatError(
            f"Unable to read statistics metadata '{metadata_path}': {error}"
        ) from error

    metadata: dict[str, str] = {}

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.partition("#")[0].strip()

        if not line:
            continue

        fields = line.split(maxsplit=1)

        if len(fields) != 2 or not fields[1].strip():
            raise _format_error(
                metadata_path,
                f"line {line_number} must contain one key and one value",
            )

        key, value = fields[0], fields[1].strip()

        if key in metadata:
            raise _format_error(
                metadata_path,
                f"line {line_number} duplicates metadata key '{key}'",
            )

        metadata[key] = value

    missing = sorted(_REQUIRED_METADATA_KEYS.difference(metadata))

    if missing:
        raise _format_error(
            metadata_path,
            "missing required metadata keys: " + ", ".join(missing),
        )

    return metadata


def _metadata_int(
    metadata: dict[str, str],
    key: str,
    path: Path,
) -> int:
    try:
        return int(metadata[key], 10)
    except ValueError as error:
        raise _format_error(
            path,
            f"metadata key '{key}' is not an integer",
        ) from error


def _metadata_float(
    metadata: dict[str, str],
    key: str,
    path: Path,
) -> float:
    try:
        value = float(metadata[key])
    except ValueError as error:
        raise _format_error(
            path,
            f"metadata key '{key}' is not a number",
        ) from error

    if not np.isfinite(value):
        raise _format_error(
            path,
            f"metadata key '{key}' must be finite",
        )

    return value


# ============================================================
# Metadata validation
# ============================================================

def _validate_metadata(
    metadata: dict[str, str],
    path: Path,
) -> np.dtype:
    if metadata["format_version"] != "2":
        raise _format_error(
            path,
            f"unsupported format_version '{metadata['format_version']}'",
        )

    if metadata["kinetic_energy_file"] != KINETIC_ENERGY_FILENAME:
        raise _format_error(
            path,
            f"kinetic_energy_file must be '{KINETIC_ENERGY_FILENAME}'",
        )

    if metadata["enstrophy_file"] != ENSTROPHY_FILENAME:
        raise _format_error(
            path,
            f"enstrophy_file must be '{ENSTROPHY_FILENAME}'",
        )

    integers = {
        key: _metadata_int(metadata, key, path)
        for key in (
            "NX",
            "NY",
            "NZ",
            "NXPhysical",
            "NYPhysical",
            "NZPhysical",
            "Nphysical",
            "Steps",
            "statisticsStart",
            "statisticsEvery",
            "record_step_offset_bytes",
            "record_value_offset_bytes",
            "record_step_bytes",
            "record_value_bytes",
            "record_bytes",
        )
    }

    for dimension in ("NX", "NY", "NZ"):
        if integers[dimension] <= 2:
            raise _format_error(
                path,
                f"{dimension} must be larger than two",
            )

    if integers["NXPhysical"] != integers["NX"] - 2:
        raise _format_error(
            path,
            "NXPhysical must equal NX - 2",
        )

    if integers["NYPhysical"] != integers["NY"] - 2:
        raise _format_error(
            path,
            "NYPhysical must equal NY - 2",
        )

    if integers["NZPhysical"] != integers["NZ"] - 2:
        raise _format_error(
            path,
            "NZPhysical must equal NZ - 2",
        )

    physical_count = (
        integers["NXPhysical"]
        * integers["NYPhysical"]
        * integers["NZPhysical"]
    )

    if integers["Nphysical"] != physical_count:
        raise _format_error(
            path,
            "Nphysical does not match the physical dimensions",
        )

    if integers["Steps"] < 0:
        raise _format_error(
            path,
            "Steps cannot be negative",
        )

    if integers["statisticsStart"] < 0:
        raise _format_error(
            path,
            "statisticsStart cannot be negative",
        )

    if integers["statisticsEvery"] <= 0:
        raise _format_error(
            path,
            "statisticsEvery must be positive",
        )

    for key in (
        "Re",
        "tgv_U0",
        "tgv_kx",
        "tgv_ky",
        "tgv_kz",
        "tgv_L",
    ):
        if _metadata_float(metadata, key, path) <= 0.0:
            raise _format_error(
                path,
                f"metadata key '{key}' must be positive",
            )

    if metadata["record_step_dtype"] != "uint32":
        raise _format_error(
            path,
            "record_step_dtype must be uint32",
        )

    if metadata["record_byte_order"] != "little_endian":
        raise _format_error(
            path,
            "record_byte_order must be little_endian",
        )

    if metadata["record_layout"] != "step_then_value":
        raise _format_error(
            path,
            "record_layout must be step_then_value",
        )

    if integers["record_step_offset_bytes"] != 0:
        raise _format_error(
            path,
            "record_step_offset_bytes must be 0",
        )

    if integers["record_value_offset_bytes"] != 4:
        raise _format_error(
            path,
            "record_value_offset_bytes must be 4",
        )

    if integers["record_step_bytes"] != 4:
        raise _format_error(
            path,
            "record_step_bytes must be 4",
        )

    # ========================================================
    # Statistics precision
    # ========================================================

    value_dtype_name = metadata["record_value_dtype"]

    if value_dtype_name == "float32":
        value_format = "<f4"
        expected_value_bytes = 4
        expected_encoding = "ieee754_binary32"

    elif value_dtype_name == "float64":
        value_format = "<f8"
        expected_value_bytes = 8
        expected_encoding = "ieee754_binary64"

    else:
        raise _format_error(
            path,
            f"unsupported record_value_dtype '{value_dtype_name}'",
        )

    if metadata["record_value_encoding"] != expected_encoding:
        raise _format_error(
            path,
            f"record_value_encoding must be '{expected_encoding}'",
        )

    if integers["record_value_bytes"] != expected_value_bytes:
        raise _format_error(
            path,
            f"record_value_bytes must be {expected_value_bytes}",
        )

    expected_record_bytes = 4 + expected_value_bytes

    if integers["record_bytes"] != expected_record_bytes:
        raise _format_error(
            path,
            f"record_bytes must be {expected_record_bytes}",
        )

    # ========================================================
    # Statistics definitions
    # ========================================================

    if metadata["kinetic_energy_definition"] != "half_velocity_squared_sum":
        raise _format_error(
            path,
            "unexpected kinetic-energy definition",
        )

    if metadata["enstrophy_definition"] != "half_vorticity_squared_sum":
        raise _format_error(
            path,
            "unexpected enstrophy definition",
        )

    if metadata["vorticity_derivative"] != "central_fourth_order":
        raise _format_error(
            path,
            "vorticity_derivative must be central_fourth_order",
        )

    grid_spacing = _metadata_float(
        metadata,
        "grid_spacing",
        path,
    )

    if not np.isclose(
        grid_spacing,
        1.0,
        rtol=0.0,
        atol=np.finfo(np.float64).eps,
    ):
        raise _format_error(
            path,
            "grid_spacing must equal 1",
        )

    dtype = np.dtype(
        {
            "names": ("step", "value"),
            "formats": ("<u4", value_format),
            "offsets": (0, 4),
            "itemsize": expected_record_bytes,
        }
    )

    return dtype


# ============================================================
# Expected sampling sequence
# ============================================================

def _expected_sample_steps(
    steps: int,
    statistics_start: int,
    statistics_every: int,
) -> np.ndarray:
    samples = [0]

    first_scheduled = (
        statistics_start
        if statistics_start > 0
        else statistics_every
    )

    if first_scheduled <= steps:
        samples.extend(
            range(
                first_scheduled,
                steps + 1,
                statistics_every,
            )
        )

    return np.asarray(
        samples,
        dtype=np.uint32,
    )


# ============================================================
# Generic scalar-history reader
# ============================================================

def _load_scalar_history(
    filename: str,
    *,
    case_name: str,
    case_directory: str | Path | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, str],
    Path,
]:
    if case_directory is None:
        resolved_case_directory = (
            VALIDATION_DIRECTORY
            / "data"
            / case_name
        )
    else:
        resolved_case_directory = Path(case_directory)

    metadata_path = (
        resolved_case_directory
        / METADATA_FILENAME
    )

    binary_path = (
        resolved_case_directory
        / filename
    )

    metadata = read_statistics_metadata(
        metadata_path
    )

    dtype = _validate_metadata(
        metadata,
        metadata_path,
    )

    try:
        binary_size = binary_path.stat().st_size
    except OSError as error:
        raise StatisticsFormatError(
            f"Unable to inspect '{binary_path}': {error}"
        ) from error

    if binary_size == 0:
        raise _format_error(
            binary_path,
            "binary history contains no records",
        )

    if binary_size % dtype.itemsize != 0:
        raise _format_error(
            binary_path,
            f"file size {binary_size} is not a multiple of "
            f"{dtype.itemsize} bytes",
        )

    expected_count = (
        binary_size
        // dtype.itemsize
    )

    try:
        records = np.fromfile(
            binary_path,
            dtype=dtype,
        )
    except OSError as error:
        raise StatisticsFormatError(
            f"Unable to read '{binary_path}': {error}"
        ) from error

    if records.size != expected_count:
        raise _format_error(
            binary_path,
            f"read {records.size} records, "
            f"expected {expected_count}",
        )

    step = records["step"].copy()

    # Convert values to float64 for Python analysis even when
    # they were stored as float32 by the simulation.
    value = records["value"].astype(
        np.float64
    )

    if step.size == 0:
        raise _format_error(
            binary_path,
            "binary history is empty",
        )

    if step[0] != 0:
        raise _format_error(
            binary_path,
            f"first sample must be step 0, "
            f"got {int(step[0])}",
        )

    if (
        step.size > 1
        and np.any(step[1:] <= step[:-1])
    ):
        raise _format_error(
            binary_path,
            "steps must be strictly increasing",
        )

    if not np.all(np.isfinite(value)):
        raise _format_error(
            binary_path,
            "history contains NaN or Inf",
        )

    if np.any(value < 0.0):
        raise _format_error(
            binary_path,
            "history contains a negative value",
        )

    steps = _metadata_int(
        metadata,
        "Steps",
        metadata_path,
    )

    statistics_start = _metadata_int(
        metadata,
        "statisticsStart",
        metadata_path,
    )

    statistics_every = _metadata_int(
        metadata,
        "statisticsEvery",
        metadata_path,
    )

    expected_steps = _expected_sample_steps(
        steps,
        statistics_start,
        statistics_every,
    )

    if not np.array_equal(
        step,
        expected_steps,
    ):
        raise _format_error(
            binary_path,
            "sample sequence does not match "
            "Steps/statisticsStart/statisticsEvery",
        )

    u0 = _metadata_float(
        metadata,
        "tgv_U0",
        metadata_path,
    )

    kx = _metadata_float(
        metadata,
        "tgv_kx",
        metadata_path,
    )

    time_star = (
        step.astype(np.float64)
        * u0
        * kx
    )

    if not np.all(
        np.isfinite(time_star)
    ):
        raise _format_error(
            binary_path,
            "derived tStar contains NaN or Inf",
        )

    if (
        time_star.size > 1
        and np.any(
            time_star[1:]
            <= time_star[:-1]
        )
    ):
        raise _format_error(
            binary_path,
            "derived tStar is not strictly increasing",
        )

    return (
        step,
        value,
        time_star,
        metadata,
        resolved_case_directory,
    )


# ============================================================
# Kinetic energy
# ============================================================

def load_kinetic_energy(
    case_name: str = DEFAULT_CASE_NAME,
    *,
    case_directory: str | Path | None = None,
) -> KineticEnergyHistory:
    (
        step,
        SK,
        tStar,
        metadata,
        directory,
    ) = _load_scalar_history(
        KINETIC_ENERGY_FILENAME,
        case_name=case_name,
        case_directory=case_directory,
    )

    metadata_path = (
        directory
        / METADATA_FILENAME
    )

    nphysical = _metadata_int(
        metadata,
        "Nphysical",
        metadata_path,
    )

    u0 = _metadata_float(
        metadata,
        "tgv_U0",
        metadata_path,
    )

    KStar = (
        SK
        / (
            float(nphysical)
            * u0
            * u0
        )
    )

    if (
        not np.all(np.isfinite(KStar))
        or np.any(KStar < 0.0)
    ):
        raise _format_error(
            directory
            / KINETIC_ENERGY_FILENAME,
            "derived KStar is invalid",
        )

    if not np.isclose(
        KStar[0],
        EXPECTED_INITIAL_KSTAR,
        rtol=INITIAL_RTOL,
        atol=np.finfo(np.float64).eps,
    ):
        relative_error = (
            abs(
                float(KStar[0])
                - EXPECTED_INITIAL_KSTAR
            )
            / EXPECTED_INITIAL_KSTAR
        )

        raise _format_error(
            directory
            / KINETIC_ENERGY_FILENAME,
            f"KStar(0)={KStar[0]:.17g} "
            f"does not match 0.125 "
            f"(relative error "
            f"{relative_error:.17g})",
        )

    return KineticEnergyHistory(
        step=step,
        SK=SK,
        tStar=tStar,
        KStar=KStar,
        metadata=metadata,
        case_directory=directory,
    )


# ============================================================
# Fourth-order enstrophy analytical value
# ============================================================

def expected_initial_enstrophy_star_fd(
    metadata: dict[str, str],
    metadata_path: Path,
) -> float:
    """
    Exact initial enstrophy expected from the fourth-order
    centered finite-difference vorticity operator.

    For

        df/dx =
        (-f[i+2] + 8 f[i+1] - 8 f[i-1] + f[i-2]) / 12

    the modified wavenumber is

        k_tilde =
        (8 sin(k) - sin(2k)) / 6.

    Therefore

        ZetaStar(0) =
        3/8 * (k_tilde/k)^2.
    """

    kx = _metadata_float(
        metadata,
        "tgv_kx",
        metadata_path,
    )

    modified_wavenumber = (
        8.0 * np.sin(kx)
        - np.sin(2.0 * kx)
    ) / 6.0

    ratio = (
        modified_wavenumber
        / kx
    )

    return (
        3.0
        / 8.0
        * ratio
        * ratio
    )


# ============================================================
# Enstrophy
# ============================================================

def load_enstrophy(
    case_name: str = DEFAULT_CASE_NAME,
    *,
    case_directory: str | Path | None = None,
) -> EnstrophyHistory:
    (
        step,
        Szeta,
        tStar,
        metadata,
        directory,
    ) = _load_scalar_history(
        ENSTROPHY_FILENAME,
        case_name=case_name,
        case_directory=case_directory,
    )

    metadata_path = (
        directory
        / METADATA_FILENAME
    )

    nphysical = _metadata_int(
        metadata,
        "Nphysical",
        metadata_path,
    )

    u0 = _metadata_float(
        metadata,
        "tgv_U0",
        metadata_path,
    )

    length = _metadata_float(
        metadata,
        "tgv_L",
        metadata_path,
    )

    normalization = (
        float(nphysical)
        * u0
        * u0
        / (
            length
            * length
        )
    )

    ZetaStar = (
        Szeta
        / normalization
    )

    if (
        not np.all(np.isfinite(ZetaStar))
        or np.any(ZetaStar < 0.0)
    ):
        raise _format_error(
            directory
            / ENSTROPHY_FILENAME,
            "derived ZetaStar is invalid",
        )

    expected_fd = (
        expected_initial_enstrophy_star_fd(
            metadata,
            metadata_path,
        )
    )

    if not np.isclose(
        ZetaStar[0],
        expected_fd,
        rtol=INITIAL_RTOL,
        atol=np.finfo(np.float64).eps,
    ):
        relative_error = (
            abs(
                float(ZetaStar[0])
                - expected_fd
            )
            / expected_fd
        )

        raise _format_error(
            directory
            / ENSTROPHY_FILENAME,
            f"ZetaStar(0)="
            f"{ZetaStar[0]:.17g} "
            f"does not match the fourth-order "
            f"centered-difference value "
            f"{expected_fd:.17g} "
            f"(relative error "
            f"{relative_error:.17g})",
        )

    return EnstrophyHistory(
        step=step,
        Szeta=Szeta,
        tStar=tStar,
        ZetaStar=ZetaStar,
        metadata=metadata,
        case_directory=directory,
    )


# ============================================================
# Summary utilities
# ============================================================

def _nearest_index(
    values: np.ndarray,
    target: float,
) -> int:
    return int(
        np.argmin(
            np.abs(
                values
                - target
            )
        )
    )


def _print_summary(
    energy: KineticEnergyHistory,
    enstrophy: EnstrophyHistory,
) -> None:
    print(
        f"case_directory: "
        f"{energy.case_directory}"
    )

    print(
        f"samples: "
        f"{energy.step.size}"
    )

    print(
        f"first_step: "
        f"{int(energy.step[0])}"
    )

    print(
        f"final_step: "
        f"{int(energy.step[-1])}"
    )

    print(
        f"final_tStar: "
        f"{energy.tStar[-1]:.17g}"
    )

    # ========================================================
    # Kinetic energy
    # ========================================================

    print()
    print("Kinetic energy")

    print(
        f"SK(0): "
        f"{energy.SK[0]:.17g}"
    )

    print(
        f"KStar(0): "
        f"{energy.KStar[0]:.17g}"
    )

    # ========================================================
    # Enstrophy
    # ========================================================

    print()
    print("Enstrophy")

    print(
        f"Szeta(0): "
        f"{enstrophy.Szeta[0]:.17g}"
    )

    print(
        f"ZetaStar(0): "
        f"{enstrophy.ZetaStar[0]:.17g}"
    )

    metadata_path = (
        energy.case_directory
        / METADATA_FILENAME
    )

    expected_fd = (
        expected_initial_enstrophy_star_fd(
            enstrophy.metadata,
            metadata_path,
        )
    )

    print(
        f"ZetaStar_FD4(0): "
        f"{expected_fd:.17g}"
    )

    print(
        "ZetaStar_continuum(0): "
        f"{EXPECTED_CONTINUUM_INITIAL_ZETASTAR:.17g}"
    )

    initial_relative_error = (
        abs(
            float(enstrophy.ZetaStar[0])
            - expected_fd
        )
        / expected_fd
    )

    print(
        "ZetaStar_FD4_relative_error: "
        f"{initial_relative_error:.17g}"
    )

    # ========================================================
    # Enstrophy peak
    # ========================================================

    peak_index = int(
        np.argmax(
            enstrophy.ZetaStar
        )
    )

    print()
    print("Enstrophy peak")

    print(
        f"step: "
        f"{int(enstrophy.step[peak_index])}"
    )

    print(
        f"tStar: "
        f"{enstrophy.tStar[peak_index]:.17g}"
    )

    print(
        f"ZetaStar: "
        f"{enstrophy.ZetaStar[peak_index]:.17g}"
    )

    # ========================================================
    # Selected samples
    # ========================================================

    print()
    print("Selected samples")

    for target in (
        0.0,
        5.0,
        9.0,
        10.0,
        15.0,
        20.0,
    ):
        index = _nearest_index(
            energy.tStar,
            target,
        )

        print(
            f"target={target:g} "
            f"tStar={energy.tStar[index]:.17g} "
            f"KStar={energy.KStar[index]:.17g} "
            f"ZetaStar={enstrophy.ZetaStar[index]:.17g} "
            f"step={int(energy.step[index])}"
        )


# ============================================================
# Command-line interface
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read and validate Taylor-Green kinetic-energy "
            "and fourth-order enstrophy statistics."
        )
    )

    parser.add_argument(
        "--case",
        default=DEFAULT_CASE_NAME,
        help=(
            "case directory name below validation/data "
            f"(default: {DEFAULT_CASE_NAME})"
        ),
    )

    parser.add_argument(
        "--case-directory",
        type=Path,
        help=(
            "explicit case directory, overriding --case"
        ),
    )

    arguments = parser.parse_args()

    try:
        energy = load_kinetic_energy(
            arguments.case,
            case_directory=arguments.case_directory,
        )

        enstrophy = load_enstrophy(
            arguments.case,
            case_directory=arguments.case_directory,
        )

    except StatisticsFormatError as error:
        parser.exit(
            status=1,
            message=f"error: {error}\n",
        )

    if not np.array_equal(
        energy.step,
        enstrophy.step,
    ):
        parser.exit(
            status=1,
            message=(
                "error: kinetic-energy and enstrophy histories "
                "do not use the same sample steps\n"
            ),
        )

    if not np.array_equal(
        energy.tStar,
        enstrophy.tStar,
    ):
        parser.exit(
            status=1,
            message=(
                "error: kinetic-energy and enstrophy histories "
                "do not use the same dimensionless times\n"
            ),
        )

    _print_summary(
        energy,
        enstrophy,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
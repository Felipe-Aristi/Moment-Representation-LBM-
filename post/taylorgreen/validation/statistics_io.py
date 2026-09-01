#!/usr/bin/env python3
"""Read and validate Taylor--Green kinetic-energy statistics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np


VALIDATION_DIRECTORY: Final = Path(__file__).resolve().parent
DEFAULT_CASE_NAME: Final = "TG_RE_1600"
KINETIC_ENERGY_FILENAME: Final = "kinetic_energy.bin"
METADATA_FILENAME: Final = "statistics_metadata.txt"
EXPECTED_INITIAL_KSTAR: Final = 0.125
INITIAL_KSTAR_RTOL: Final = 1.0e-5

_REQUIRED_METADATA_KEYS: Final = frozenset(
    {
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
        "tgv_L",
        "statisticsStart",
        "statisticsEvery",
        "record_step_dtype",
        "record_SK_dtype",
        "record_byte_order",
        "record_layout",
        "record_step_offset_bytes",
        "record_SK_offset_bytes",
        "record_step_bytes",
        "record_SK_bytes",
        "record_SK_encoding",
        "record_bytes",
    }
)


class StatisticsFormatError(ValueError):
    """Raised when statistics metadata or binary data is inconsistent."""


@dataclass(frozen=True)
class KineticEnergyHistory:
    """Validated raw and dimensionless Taylor--Green kinetic-energy history."""

    step: np.ndarray
    SK: np.ndarray
    tStar: np.ndarray
    KStar: np.ndarray
    metadata: dict[str, str]
    case_directory: Path


def _format_error(path: Path, message: str) -> StatisticsFormatError:
    return StatisticsFormatError(f"{path}: {message}")


def read_statistics_metadata(path: str | Path) -> dict[str, str]:
    """Read the whitespace-separated ``key value`` metadata format."""

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

    missing_keys = sorted(_REQUIRED_METADATA_KEYS.difference(metadata))

    if missing_keys:
        raise _format_error(
            metadata_path,
            "missing required metadata keys: " + ", ".join(missing_keys),
        )

    return metadata


def _metadata_int(metadata: dict[str, str], key: str, path: Path) -> int:
    try:
        return int(metadata[key], 10)
    except ValueError as error:
        raise _format_error(path, f"metadata key '{key}' is not an integer") from error


def _metadata_float(metadata: dict[str, str], key: str, path: Path) -> float:
    try:
        value = float(metadata[key])
    except ValueError as error:
        raise _format_error(path, f"metadata key '{key}' is not a number") from error

    if not np.isfinite(value):
        raise _format_error(path, f"metadata key '{key}' must be finite")

    return value


def _validate_metadata(metadata: dict[str, str], path: Path) -> np.dtype:
    integer_values = {
        key: _metadata_int(metadata, key, path)
        for key in (
            "NX",
            "NY",
            "NZ",
            "NXPhysical",
            "NYPhysical",
            "NZPhysical",
            "Nphysical",
            "statisticsStart",
            "statisticsEvery",
            "record_step_offset_bytes",
            "record_SK_offset_bytes",
            "record_step_bytes",
            "record_SK_bytes",
            "record_bytes",
        )
    }

    for key in ("Re", "tgv_U0", "tgv_kx", "tgv_L"):
        value = _metadata_float(metadata, key, path)

        if value <= 0.0:
            raise _format_error(path, f"metadata key '{key}' must be positive")

    for dimension in ("NX", "NY", "NZ"):
        if integer_values[dimension] <= 2:
            raise _format_error(path, f"metadata key '{dimension}' must exceed two")

    physical_dimensions = (
        ("NX", "NXPhysical"),
        ("NY", "NYPhysical"),
        ("NZ", "NZPhysical"),
    )

    for stored_key, physical_key in physical_dimensions:
        if integer_values[physical_key] != integer_values[stored_key] - 2:
            raise _format_error(
                path,
                f"metadata key '{physical_key}' must equal {stored_key} - 2",
            )

    physical_count = (
        integer_values["NXPhysical"]
        * integer_values["NYPhysical"]
        * integer_values["NZPhysical"]
    )

    if integer_values["Nphysical"] != physical_count:
        raise _format_error(
            path,
            "metadata key 'Nphysical' does not match the physical dimensions",
        )

    if integer_values["statisticsStart"] < 0:
        raise _format_error(path, "metadata key 'statisticsStart' cannot be negative")

    if integer_values["statisticsEvery"] <= 0:
        raise _format_error(path, "metadata key 'statisticsEvery' must be positive")

    expected_text_values = {
        "record_step_dtype": "uint32",
        "record_SK_dtype": "float64",
        "record_byte_order": "little_endian",
        "record_layout": "step_then_SK",
        "record_SK_encoding": "ieee754_binary64",
    }

    for key, expected_value in expected_text_values.items():
        if metadata[key] != expected_value:
            raise _format_error(
                path,
                f"metadata key '{key}' must be '{expected_value}', got '{metadata[key]}'",
            )

    expected_integer_values = {
        "record_step_offset_bytes": 0,
        "record_SK_offset_bytes": 4,
        "record_step_bytes": 4,
        "record_SK_bytes": 8,
        "record_bytes": 12,
    }

    for key, expected_value in expected_integer_values.items():
        if integer_values[key] != expected_value:
            raise _format_error(
                path,
                f"metadata key '{key}' must be {expected_value}, got {integer_values[key]}",
            )

    if "format_version" in metadata and metadata["format_version"] != "1":
        raise _format_error(path, "unsupported statistics format version")

    if "statistics_file" in metadata and metadata["statistics_file"] != KINETIC_ENERGY_FILENAME:
        raise _format_error(
            path,
            f"metadata key 'statistics_file' must be '{KINETIC_ENERGY_FILENAME}'",
        )

    record_dtype = np.dtype(
        {
            "names": ("step", "SK"),
            "formats": ("<u4", "<f8"),
            "offsets": (0, 4),
            "itemsize": 12,
        }
    )

    if (
        record_dtype.itemsize != integer_values["record_bytes"]
        or record_dtype.fields["step"][1] != integer_values["record_step_offset_bytes"]
        or record_dtype.fields["SK"][1] != integer_values["record_SK_offset_bytes"]
    ):
        raise _format_error(path, "NumPy record layout does not match the metadata")

    return record_dtype


def _first_true_index(mask: np.ndarray) -> int:
    return int(np.flatnonzero(mask)[0])


def load_kinetic_energy(
    case_name: str = DEFAULT_CASE_NAME,
    *,
    case_directory: str | Path | None = None,
) -> KineticEnergyHistory:
    """Load, validate, and nondimensionalize one TGV kinetic-energy history."""

    if case_directory is None:
        resolved_case_directory = VALIDATION_DIRECTORY / "data" / case_name
    else:
        resolved_case_directory = Path(case_directory)

    metadata_path = resolved_case_directory / METADATA_FILENAME
    binary_path = resolved_case_directory / KINETIC_ENERGY_FILENAME
    metadata = read_statistics_metadata(metadata_path)
    record_dtype = _validate_metadata(metadata, metadata_path)

    try:
        binary_size = binary_path.stat().st_size
    except OSError as error:
        raise StatisticsFormatError(
            f"Unable to inspect kinetic-energy data '{binary_path}': {error}"
        ) from error

    if binary_size == 0:
        raise _format_error(binary_path, "binary history contains no records")

    if binary_size % record_dtype.itemsize != 0:
        raise _format_error(
            binary_path,
            f"file size {binary_size} is not a multiple of the {record_dtype.itemsize}-byte record size",
        )

    expected_records = binary_size // record_dtype.itemsize

    try:
        records = np.fromfile(binary_path, dtype=record_dtype)
    except OSError as error:
        raise StatisticsFormatError(
            f"Unable to read kinetic-energy data '{binary_path}': {error}"
        ) from error

    if records.size != expected_records:
        raise _format_error(
            binary_path,
            f"read {records.size} records, expected {expected_records}",
        )

    step = records["step"].copy()
    kinetic_energy_sum = records["SK"].copy()

    if step[0] != 0:
        raise _format_error(binary_path, f"first sample must be step 0, got {int(step[0])}")

    if step.size > 1:
        nonincreasing_steps = step[1:] <= step[:-1]

        if np.any(nonincreasing_steps):
            index = _first_true_index(nonincreasing_steps) + 1
            raise _format_error(
                binary_path,
                f"steps must be strictly increasing; invalid sample at index {index}",
            )

    nonfinite_energy = ~np.isfinite(kinetic_energy_sum)

    if np.any(nonfinite_energy):
        index = _first_true_index(nonfinite_energy)
        raise _format_error(binary_path, f"SK is not finite at sample index {index}")

    negative_energy = kinetic_energy_sum < 0.0

    if np.any(negative_energy):
        index = _first_true_index(negative_energy)
        raise _format_error(binary_path, f"SK is negative at sample index {index}")

    statistics_start = _metadata_int(metadata, "statisticsStart", metadata_path)
    statistics_every = _metadata_int(metadata, "statisticsEvery", metadata_path)
    scheduled_steps = step[1:]

    if scheduled_steps.size:
        if statistics_start > 0:
            sample_due = (scheduled_steps >= statistics_start) & (
                (scheduled_steps - statistics_start) % statistics_every == 0
            )
        else:
            sample_due = scheduled_steps % statistics_every == 0

        if not np.all(sample_due):
            index = _first_true_index(~sample_due) + 1
            raise _format_error(
                binary_path,
                f"step {int(step[index])} at sample index {index} is inconsistent with the sampling schedule",
            )

    u0 = _metadata_float(metadata, "tgv_U0", metadata_path)
    kx = _metadata_float(metadata, "tgv_kx", metadata_path)
    physical_count = _metadata_int(metadata, "Nphysical", metadata_path)
    step_float = step.astype(np.float64)
    time_star = step_float * u0 * kx
    energy_star = kinetic_energy_sum / (float(physical_count) * u0 * u0)

    if not np.all(np.isfinite(time_star)):
        raise _format_error(binary_path, "derived tStar contains a non-finite value")

    if time_star.size > 1:
        nonincreasing_time = time_star[1:] <= time_star[:-1]

        if np.any(nonincreasing_time):
            index = _first_true_index(nonincreasing_time) + 1
            raise _format_error(
                binary_path,
                f"derived tStar must be strictly increasing; invalid sample at index {index}",
            )

    invalid_energy_star = (~np.isfinite(energy_star)) | (energy_star < 0.0)

    if np.any(invalid_energy_star):
        index = _first_true_index(invalid_energy_star)
        raise _format_error(
            binary_path,
            f"derived KStar is non-finite or negative at sample index {index}",
        )

    if not np.isclose(
        energy_star[0],
        EXPECTED_INITIAL_KSTAR,
        rtol=INITIAL_KSTAR_RTOL,
        atol=np.finfo(np.float64).eps,
    ):
        relative_error = abs(float(energy_star[0]) - EXPECTED_INITIAL_KSTAR) / EXPECTED_INITIAL_KSTAR
        raise _format_error(
            binary_path,
            f"KStar(0)={energy_star[0]:.17g} is inconsistent with 0.125 "
            f"(relative error {relative_error:.17g})",
        )

    return KineticEnergyHistory(
        step=step,
        SK=kinetic_energy_sum,
        tStar=time_star,
        KStar=energy_star,
        metadata=metadata,
        case_directory=resolved_case_directory,
    )


def _print_summary(history: KineticEnergyHistory) -> None:
    binary_path = history.case_directory / KINETIC_ENERGY_FILENAME
    initial_relative_error = (
        abs(float(history.KStar[0]) - EXPECTED_INITIAL_KSTAR) / EXPECTED_INITIAL_KSTAR
    )

    print(f"case_directory: {history.case_directory}")
    print(f"samples: {history.step.size}")
    print(f"binary_bytes: {binary_path.stat().st_size}")
    print(f"first_step: {int(history.step[0])}")
    print(f"final_step: {int(history.step[-1])}")
    print(f"final_tStar: {history.tStar[-1]:.17g}")
    print(f"SK(0): {history.SK[0]:.17g}")
    print(f"KStar(0): {history.KStar[0]:.17g}")
    print(f"KStar(0)_relative_error: {initial_relative_error:.17g}")

    for target in (0.0, 5.0, 9.0, 10.0, 15.0, 20.0):
        index = int(np.argmin(np.abs(history.tStar - target)))
        print(
            f"nearest_tStar({target:g}): {history.tStar[index]:.17g} "
            f"KStar={history.KStar[index]:.17g} step={int(history.step[index])}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read and validate Taylor-Green kinetic-energy statistics."
    )
    parser.add_argument(
        "--case",
        default=DEFAULT_CASE_NAME,
        help=f"case directory name below validation/data (default: {DEFAULT_CASE_NAME})",
    )
    parser.add_argument(
        "--case-directory",
        type=Path,
        help="explicit case data directory, overriding --case",
    )
    arguments = parser.parse_args()

    try:
        history = load_kinetic_energy(
            arguments.case,
            case_directory=arguments.case_directory,
        )
    except StatisticsFormatError as error:
        parser.exit(status=1, message=f"error: {error}\n")

    _print_summary(history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

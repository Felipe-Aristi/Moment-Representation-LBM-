
#!/usr/bin/env python3
"""Read MR-LBM turbulent-jet radial statistics (FP32 or FP64 binary sums)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import warnings

import numpy as np


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / 'data' / 'Re5000' / 'mean_profiles'

RADIAL_MOMENTS = ('uy', 'uy2', 'ur', 'ur2', 'uruy')


# ============================================================
# Radial profiles
# ============================================================

@dataclass
class RadialProfiles:
    uy: np.ndarray
    ur: np.ndarray
    uy_rms: np.ndarray
    ur_rms: np.ndarray
    reynolds_uruy: np.ndarray
    count: np.ndarray
    metadata: dict[str, str]


# ============================================================
# Read metadata
# ============================================================

def read_metadata(folder: str | Path) -> dict[str, str]:
    path = Path(folder) / 'radial_profile_metadata.txt'

    if not path.is_file():
        raise FileNotFoundError(f'Missing jet statistics metadata: {path}')

    result: dict[str, str] = {}

    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.partition('#')[0].strip()

        if not line:
            continue

        parts = line.split(maxsplit=1)

        if len(parts) != 2 or parts[0] in result:
            raise ValueError(f'Invalid or repeated metadata field: {line}')

        result[parts[0]] = parts[1]

    return result


# ============================================================
# Read binary array
# ============================================================

def _read_array(
    folder: Path,
    filename: str,
    dtype: str,
    shape: tuple[int, int],
) -> np.ndarray:

    path = folder / filename

    if not path.is_file():
        raise FileNotFoundError(f'Missing radial statistics file: {path}')

    expected = shape[0] * shape[1]
    data_type = np.dtype(dtype)

    expected_bytes = expected * data_type.itemsize
    actual_bytes = path.stat().st_size

    if actual_bytes != expected_bytes:
        raise ValueError(
            f'{path}: found {actual_bytes} bytes, expected {expected_bytes} '
            f'for {expected} {data_type.name} elements'
        )

    array = np.fromfile(path, dtype=data_type)

    if array.size != expected:
        raise ValueError(
            f'{path}: found {array.size} elements, expected {expected}'
        )

    return array.reshape(shape)


# ============================================================
# Detect binary precision
# ============================================================

def _detect_sum_dtype(
    folder: Path,
    shape: tuple[int, int],
    metadata: dict[str, str],
) -> str:

    path = folder / 'radial_sum_uy.bin'

    if not path.is_file():
        raise FileNotFoundError(f'Missing radial statistics file: {path}')

    elements = shape[0] * shape[1]
    size = path.stat().st_size

    if size == elements * 4:
        dtype = '<f4'
        detected = 'float32'

    elif size == elements * 8:
        dtype = '<f8'
        detected = 'float64'

    else:
        raise ValueError(
            f'{path}: unexpected size {size} bytes. Expected '
            f'{elements * 4} bytes (FP32) or {elements * 8} bytes (FP64). '
            'If the solver is writing files, retry after the next output finishes.'
        )

    declared = metadata.get('stat_dtype')

    if declared is not None and declared != detected:
        warnings.warn(
            f'Metadata declares stat_dtype {declared}, but binary file size '
            f'indicates {detected}. Reading the actual file format. '
            'Correct the CUDA metadata writer.',
            RuntimeWarning,
            stacklevel=2,
        )

    return dtype


# ============================================================
# Load radial statistics
# ============================================================

def load_radial_profiles(
    folder: str | Path = DEFAULT_DATA_DIR,
) -> RadialProfiles:

    folder = Path(folder)
    metadata = read_metadata(folder)

    if metadata.get('count_dtype', 'uint64') != 'uint64':
        raise ValueError('Expected uint64 radial counts')

    ny = int(metadata['NY'])
    nr = int(metadata['NR_BINS'])

    if ny <= 0 or nr <= 0:
        raise ValueError('Nonpositive NY or NR_BINS in metadata')

    shape = (ny, nr)

    # Detect actual binary precision
    sum_dtype = _detect_sum_dtype(folder, shape, metadata)

    # Read radial moments
    sums = {
        name: _read_array(
            folder,
            f'radial_sum_{name}.bin',
            sum_dtype,
            shape,
        )
        for name in RADIAL_MOMENTS
    }

    # Read sample counts
    count = _read_array(
        folder,
        'radial_count.bin',
        '<u8',
        shape,
    )

    # Validate data
    if any(not np.isfinite(values).all() for values in sums.values()):
        raise ValueError('Non-finite radial sums')

    if not np.any(count > 0):
        raise ValueError(
            'No radial samples: increase Steps or reduce '
            'jetStatsStartTStar for testing'
        )

    # ========================================================
    # Recover mean velocity moments
    # ========================================================

    counts = count.astype(np.float64)

    def mean(values: np.ndarray) -> np.ndarray:
        out = np.full(shape, np.nan, dtype=np.float64)

        np.divide(
            values.astype(np.float64),
            counts,
            out=out,
            where=count > 0,
        )

        return out

    uy = mean(sums['uy'])
    ur = mean(sums['ur'])

    uy2 = mean(sums['uy2'])
    ur2 = mean(sums['ur2'])
    uruy = mean(sums['uruy'])

    # ========================================================
    # RMS velocities
    # ========================================================

    uy_rms = np.sqrt(
        np.maximum(uy2 - uy * uy, 0.0)
    )

    ur_rms = np.sqrt(
        np.maximum(ur2 - ur * ur, 0.0)
    )

    # ========================================================
    # Reynolds shear stress
    # ========================================================

    reynolds_uruy = uruy - ur * uy

    return RadialProfiles(
        uy=uy,
        ur=ur,
        uy_rms=uy_rms,
        ur_rms=ur_rms,
        reynolds_uruy=reynolds_uruy,
        count=count,
        metadata=metadata,
    )


# ============================================================
# Fit jet centerline decay and virtual origin
# ============================================================

def fit_virtual_origin(
    y: np.ndarray,
    uc: np.ndarray,
    diameter: float,
    inlet_velocity: float,
    fit_start: float = 10.0,
    fit_end: float = 15.0,
) -> tuple[float, float, np.ndarray, np.ndarray]:

    """
    Fit:

        U_jet / U_c = (y - y0) / (B * D)

    Returns:
        B, y0, valid y/D, fit values
    """

    if diameter <= 0 or inlet_velocity <= 0 or fit_start >= fit_end:
        raise ValueError(
            'Require positive diameter/velocity and fit_start < fit_end'
        )

    y = np.asarray(y, dtype=np.float64)
    uc = np.asarray(uc, dtype=np.float64)

    valid = (
        np.isfinite(uc)
        & (uc > 0)
        & (y / diameter >= fit_start)
        & (y / diameter <= fit_end)
    )

    if np.count_nonzero(valid) < 2:
        raise ValueError(
            f'Need >=2 valid centerline values between '
            f'{fit_start}D and {fit_end}D'
        )

    x = y[valid] / diameter
    values = inlet_velocity / uc[valid]

    slope, intercept = np.polyfit(x, values, 1)

    if not np.isfinite(slope) or slope <= 0:
        raise ValueError(
            f'Nonpositive centerline-decay slope {slope}; '
            'inspect sampled jet statistics'
        )

    B = 1.0 / slope
    y0 = -intercept * B * diameter

    return B, y0, x, values


# ============================================================
# Command-line arguments
# ============================================================

def parse_args(description: str) -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '--data',
        type=Path,
        default=DEFAULT_DATA_DIR,
        help='Directory containing radial_sum_*.bin and metadata',
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).resolve().parent / 'plots',
    )

    parser.add_argument('--fit-start', type=float, default=10.0)
    parser.add_argument('--fit-end', type=float, default=15.0)

    parser.add_argument(
        '--slices',
        nargs='+',
        type=float,
        default=[10, 11, 12, 13, 14, 15],
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':

    args = parse_args('Inspect MR-LBM jet radial statistics')

    profile = load_radial_profiles(args.data)

    expected = (
        int(profile.metadata['NY'])
        * int(profile.metadata['NR_BINS'])
    )

    file_size = (args.data / 'radial_sum_uy.bin').stat().st_size

    detected_precision = (
        'float32' if file_size == expected * 4 else 'float64'
    )

    print('Data:', args.data)
    print('Stored radial precision:', detected_precision)
    print('Samples:', profile.metadata.get('radial_sample_count', 'unknown'))
    print('Total radial count:', int(profile.count.sum()))
    print(
        'Mean axial velocity at first downstream ring:',
        profile.uy[0, 0],
    )

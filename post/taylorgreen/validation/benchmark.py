"""Official HiOCFD C3.5 spectral reference, Re=1600, 512 cubed."""

from pathlib import Path

import numpy as np


REFERENCE_LABEL = r"HiOCFD spectral DNS ($512^3$)"
REFERENCE_PATH = Path(__file__).resolve().parent / "reference" / "spectral_Re1600_512.gdiag"


def load_benchmark(metadata: dict[str, str]) -> np.ndarray:
    """Return columns t*, K*, epsilon*, zeta*, without rescaling."""
    if not np.isclose(float(metadata["Re"]), 1600.0, rtol=1e-8, atol=0.0):
        raise ValueError("The bundled HiOCFD benchmark requires Re=1600")
    wave_numbers = [float(metadata[key]) for key in ("tgv_kx", "tgv_ky", "tgv_kz")]
    if not np.allclose(wave_numbers, wave_numbers[0], rtol=1e-6, atol=0.0):
        raise ValueError("The HiOCFD benchmark requires equal wavelengths in all directions")
    data = np.loadtxt(REFERENCE_PATH)
    if data.ndim != 2 or data.shape[1] != 4 or not np.all(np.isfinite(data)):
        raise ValueError(f"Invalid benchmark table: {REFERENCE_PATH}")
    if not np.all(np.diff(data[:, 0]) > 0):
        raise ValueError("Benchmark times must be strictly increasing")
    return data

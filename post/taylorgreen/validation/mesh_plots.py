"""Discover Taylor–Green mesh runs and plot comparisons with the reference."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark import REFERENCE_LABEL, load_benchmark
from statistics_io import load_kinetic_energy, load_enstrophy, read_statistics_metadata

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parents[1]))
from prettyPlot import pretty_plot


def discover_cases(data_directory=None):
    """Use metadata to identify meshes; prefer named runs over legacy copies."""
    data_directory = Path(data_directory) if data_directory is not None else DIRECTORY / "data"
    cases = {}
    for path in sorted(data_directory.glob("TG_RE_*/statistics_metadata.txt")):
        metadata = read_statistics_metadata(path)
        if not np.isclose(float(metadata["Re"]), 1600.0):
            continue
        histories = [path.parent / name for name in ("kinetic_energy.bin", "enstrophy.bin")]
        if any(not history.exists() or history.stat().st_size == 0 for history in histories):
            print(f"Skipping {path.parent.name}: statistics files are missing or empty")
            continue
        mesh = tuple(int(metadata[key]) for key in ("NXPhysical", "NYPhysical", "NZPhysical"))
        key = (float(metadata["Re"]), mesh)
        named = path.parent.name.endswith("_" + "x".join(map(str, mesh)))
        if key not in cases or named:
            cases[key] = path.parent
    if not cases:
        raise FileNotFoundError(f"No Re=1600 Taylor–Green statistics found in {data_directory}")
    return [cases[key] for key in sorted(cases)]


def plot_mesh_comparison(quantity):
    curves = []
    reference = None
    for directory in discover_cases():
        energy = load_kinetic_energy(case_directory=directory)
        reference = load_benchmark(energy.metadata)
        mesh = tuple(int(energy.metadata[key]) for key in ("NXPhysical", "NYPhysical", "NZPhysical"))
        label = f"MR-RR ${mesh[0]}^3$" if len(set(mesh)) == 1 else "MR-RR " + "x".join(map(str, mesh))
        if quantity == "energy":
            curves.append((energy.tStar, energy.KStar, label, "-"))
        else:
            enstrophy = load_enstrophy(case_directory=directory)
            if quantity == "enstrophy":
                curves.append((enstrophy.tStar, enstrophy.ZetaStar, label, "-"))
            elif quantity == "dissipation_energy":
                if not np.array_equal(energy.step, enstrophy.step) or not np.array_equal(energy.tStar, enstrophy.tStar):
                    raise ValueError(f"Mismatched energy/enstrophy samples in {directory}")
                if energy.tStar.size < 3:
                    raise ValueError(f"At least three samples needed for dissipation in {directory}")
                curves.append((energy.tStar, -np.gradient(energy.KStar, energy.tStar, edge_order=2), label, "-"))
            elif quantity == "dissipation_enstrophy":
                curves.append((enstrophy.tStar, 2 * enstrophy.ZetaStar / float(energy.metadata["Re"]), label, "-"))
        print(f"Loaded {directory.name}: physical mesh {mesh}")

    column, ylabel, filename = {
        "energy": (1, r"$K^*$", "kinetic_energy.png"),
        "enstrophy": (3, r"$\zeta^*$", "enstrophy.png"),
        "dissipation_energy": (2, r"$-\mathrm{d}K^*/\mathrm{d}t^*$", "dissipation_energy.png"),
        "dissipation_enstrophy": (3, r"$2\zeta^*/Re$", "dissipation_enstrophy.png"),
    }[quantity]
    reference_values = reference[:, column]
    if quantity == "dissipation_enstrophy":
        reference_values = 2 * reference_values / 1600.0
    values = np.concatenate([curve[1] for curve in curves] + [reference_values])
    if not np.all(np.isfinite(values)):
        raise ValueError("Cannot plot non-finite statistics")
    lower, upper = float(values.min()), float(values.max())
    padding = max(.06 * (upper - lower), 1e-6)
    fig, ax, _ = pretty_plot(
        xLim=(0., 20.), yLim=(max(0., lower - padding) if lower >= 0 else lower - padding, upper + padding),
        nxTicks=6, nyTicks=6, xLabel=r"$t^*$", yLabel=ylabel,
        xTickFormat=0, yTickFormat=3 if quantity != "enstrophy" else 2,
        useGrid=True, plotAspectRatio=(1.5, 1., 1.),
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (time, values, label, style) in enumerate(curves):
        color_index = i
        ax.plot(time, values, color=colors[color_index % len(colors)], linestyle=style, linewidth=1.25, label=label)
    ax.plot(reference[:, 0], reference_values, color="black", linestyle=":", linewidth=1.5, label=REFERENCE_LABEL)
    ax.legend(frameon=False, loc="best", fontsize="small")
    output = DIRECTORY / "plots" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output

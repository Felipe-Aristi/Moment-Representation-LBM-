#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np


VALIDATION_DIRECTORY = Path(__file__).resolve().parent

sys.path.insert(0, str(VALIDATION_DIRECTORY))

from post.channelflow.validation.statistics_io import load_kinetic_energy, load_enstrophy


def derivative_fourth_order(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    if values.size < 5:
        raise ValueError("At least five samples are required")

    dx = np.diff(x)

    if not np.allclose(dx, dx[0], rtol=1.0e-10, atol=1.0e-14):
        raise ValueError("Fourth-order derivative requires uniform sampling")

    h = float(dx[0])

    derivative = np.gradient(
        values,
        x,
        edge_order=2,
    )

    derivative[2:-2] = (
        values[:-4]
        - 8.0 * values[1:-3]
        + 8.0 * values[3:-1]
        - values[4:]
    ) / (12.0 * h)

    return derivative


def main() -> None:
    energy = load_kinetic_energy()
    enstrophy = load_enstrophy()

    if not np.array_equal(energy.step, enstrophy.step):
        raise ValueError("Statistics histories use different steps")

    t_star = energy.tStar
    k_star = energy.KStar
    zeta_star = enstrophy.ZetaStar

    reynolds = float(energy.metadata["Re"])

    epsilon_second = -np.gradient(
        k_star,
        t_star,
        edge_order=2,
    )

    epsilon_fourth = -derivative_fourth_order(
        k_star,
        t_star,
    )

    epsilon_enstrophy = (
        2.0 * zeta_star / reynolds
    )

    peak_second = int(np.argmax(epsilon_second))
    peak_fourth = int(np.argmax(epsilon_fourth))
    peak_enstrophy = int(np.argmax(epsilon_enstrophy))

    print("Second-order time derivative")
    print(f"tStar       = {t_star[peak_second]:.17g}")
    print(f"epsilonStar = {epsilon_second[peak_second]:.17g}")

    print()
    print("Fourth-order time derivative")
    print(f"tStar       = {t_star[peak_fourth]:.17g}")
    print(f"epsilonStar = {epsilon_fourth[peak_fourth]:.17g}")

    print()
    print("Enstrophy-based")
    print(f"tStar       = {t_star[peak_enstrophy]:.17g}")
    print(f"epsilonStar = {epsilon_enstrophy[peak_enstrophy]:.17g}")

    relative_peak_change = abs(
        epsilon_fourth[peak_fourth]
        - epsilon_second[peak_second]
    ) / epsilon_fourth[peak_fourth]

    print()
    print(
        "2nd-vs-4th time-derivative peak difference = "
        f"{relative_peak_change:.17g}"
    )

    # Compare the two physical estimates at every common time.
    valid = epsilon_enstrophy > np.finfo(np.float64).eps

    ratio = np.full_like(
        epsilon_fourth,
        np.nan,
        dtype=np.float64,
    )

    ratio[valid] = (
        epsilon_fourth[valid]
        / epsilon_enstrophy[valid]
    )

    print()
    print("Dissipation ratio epsilon_K / epsilon_zeta")

    for target in (0.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0):
        index = int(np.argmin(np.abs(t_star - target)))

        print(
            f"target={target:4.1f} "
            f"tStar={t_star[index]:.8f} "
            f"epsK={epsilon_fourth[index]:.8e} "
            f"epsZeta={epsilon_enstrophy[index]:.8e} "
            f"ratio={ratio[index]:.8f}"
        )


if __name__ == "__main__":
    main()
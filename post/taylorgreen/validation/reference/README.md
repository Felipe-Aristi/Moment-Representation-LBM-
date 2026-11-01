# Official Taylor–Green vortex reference

`spectral_Re1600_512.gdiag` is copied unchanged from the reference archive
linked under case C3.5 on the first International Workshop on High-Order CFD
Methods website:

- Workshop: https://cfd.ku.edu/hiocfd.html
- Reference archive: https://cfd.ku.edu/hiocfd/data.tgz
- Case specification: https://cfd.ku.edu/hiocfd/case_c3.5.pdf
- Retrieved: 2026-11-01
- SHA-256: `8e01c0024ae2dde20df8dc77a20fa7fb41c3abef2fc528a2ac684afed96d6429`

The file identifies its source as a 512 x 512 x 512 spectral code. It has
four columns: dimensionless time, kinetic energy, dissipation (-dE/dt), and
enstrophy. It covers t*=0 through 19.99. No interpolation, digitization,
or fitted curve was used to create this reference file.

The normalization matches this project's statistics:

- t* = t U0 / L, where the periodic box has side 2 pi L.
- K* = <|u|^2> / (2 U0^2), initially 0.125.
- zeta* = <|curl u|^2> L^2 / (2 U0^2), initially 0.375.
- epsilon* = -dK*/dt*, approximately 2 zeta*/1600.

Use the supplied dissipation column directly; do not differentiate the
reference energy or renormalize it by its initial value. The project energy
is velocity based; density-weighted compressible energy definitions can differ.
The overlay requires Re=1600 and equal wavelengths in all three directions.
It is a reference comparison, not an automatic pass/fail validation.

From the repository root, regenerate the plots with:

```sh
python3 post/taylorgreen/validation/plot_energy.py
python3 post/taylorgreen/validation/plot_enstrophy.py
python3 post/taylorgreen/validation/plot_dissipation.py
```

Simulation histories are read from `data/TG_RE_1600/`; plots are written
to `plots/`. Both paths are relative to the validation directory.
Reference curves are black dotted lines; simulation curves retain their styles.

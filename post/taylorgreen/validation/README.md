# Taylor–Green mesh comparisons

The solver automatically names statistics folders using the physical grid:
`data/TG_RE_1600_64x64x64/`. VTI output uses the same case name under
`post/taylorgreen/`. Ghost cells are excluded from the name.
Repeating the same Reynolds number and mesh replaces that run's statistics.

Set all three dimensions in `src/constants.cuh`, then rebuild and run using
`bash run.sh` from the repository root:

| Physical mesh | NX = NY = NZ | Steps for approximately t*=20 at U0=0.05 |
| --- | --- | --- |
| 64^3 | 66 | 4075 |
| 128^3 | 130 | 8149 |
| 256^3 | 258 | 16298 |
| 512^3 | 514 | 32595 |

The required step count is `ceil(20 * tgv_L / tgv_U0)`, with
`tgv_L = NXPhysical / (2*pi)`. Increase `Steps` when refining so every run
covers the same dimensionless time. Viscosity follows the configured Re
automatically. This does not launch the expensive mesh runs automatically.

After any completed run, execute:

```sh
python3 post/taylorgreen/validation/plot_energy.py
python3 post/taylorgreen/validation/plot_enstrophy.py
python3 post/taylorgreen/validation/plot_dissipation.py
```

Each script discovers all Re=1600 cases under `validation/data`, reads mesh
sizes from metadata, and overlays the available runs with the official
spectral reference. Dissipation is split into `dissipation_energy.png` (-dK*/dt*) and
`dissipation_enstrophy.png` (2 zeta*/Re), with matching mesh colors.
Here zeta* is dimensionless enstrophy, half the mean squared vorticity
scaled by L^2/U0^2. The latter estimates viscous dissipation; the two
expressions agree for resolved incompressible periodic flow. Plots are saved under `validation/plots`.

The old `data/TG_RE_1600/` folder is supported too. When a mesh-named folder
and the old folder describe the same mesh, the mesh-named folder takes priority.
Malformed metadata or incomplete statistics cause an error rather than silently
disappearing from the comparison. Only Re=1600 is plotted against this reference.

To inspect an individual mesh using the statistics checker:

```sh
python3 post/taylorgreen/validation/statistics_io.py --case TG_RE_1600_128x128x128
```

Reference provenance and normalization are documented in [reference/README.md](reference/README.md).

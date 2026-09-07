# Low-Mach resistive MHD: Orszag–Tang

This case couples the existing D3Q27 moment-based R3 fluid solver to a
D3Q7 vector BGK induction solver. It runs the planar Orszag–Tang vortex
with periodic boundaries in all three directions.

## Run

From the repository root:

```bash
bash run.sh mhd
```

The default target is the RTX 4070 (`sm_89`), with CUDA and C++20.
Use `CUDA_ARCH=sm_XX` or `NVCC=/path/to/nvcc` for another CUDA installation.
The build compiles only `cases/MHD/main.cu` and `cases/MHD/kernels.cu`;
shared fluid reconstruction and memory utilities come from `src/`.

For a short GPU check in a new output directory:

```bash
MHD_NVCC_FLAGS="-DMHD_NX=32 -DMHD_STEPS=20" bash run.sh mhd 0 output/MHD/smoke
```

To compile without running:

```bash
BUILD_ONLY=1 bash run.sh mhd
```

For subsequent runs, pass a new directory:
`bash run.sh mhd 0 output/MHD/run_02`. Existing nonempty output directories
are rejected to protect simulation results. The driver also accepts
`bash run.sh taylorgreen`; running it without arguments still selects Taylor–Green.

## Parameters and initial conditions

Edit `config.cuh` and recompile. Grid dimensions, stencil operations,
relaxation parameters, and population loops are determined at compile time.
Supported compiler overrides are `MHD_NX`, `MHD_NY`, `MHD_NZ`,
`MHD_RE`, `MHD_RM`, and `MHD_STEPS`.

| Quantity | Default |
|---|---|
| Physical grid | 128 × 128 × 1 |
| Stored grid, including ghost planes | 130 × 130 × 3 |
| U0, B0, reference density | 0.02, 0.02, 1 |
| Re, Rm | 100, 100 |
| Reference length L | Nx / (2π) |
| Kinematic viscosity ν | U0 L / Re |
| Magnetic diffusivity η | U0 L / Rm |
| Magnetic Prandtl number | ν / η = Rm / Re |
| Steps | 10,000 (final t* ≈ 9.8175) |
| Statistics / field output interval | 50 / 500 steps |

In lattice units Δx = Δt = μ0 = 1. With X = 2πx/Nx and Y = 2πy/Ny,
where x,y start at zero on the physical interior:

```text
u = U0 (-sin Y, sin X, 0)
B = B0 (-sin Y, sin 2X, 0)
t* = t U0 / L
```

The configuration requires Nx = Ny. Increasing Nz repeats this same planar
initial condition; a three-dimensional initial perturbation is a separate choice.

The initial density is rho0 + p'/cs², using the leading incompressible
pressure Poisson solution:

```text
p' = -rho0 U0² cos X cos Y + (4/5) B0² cos 2X cos Y
     - (|B|² - B0²)/2.
```

The initial kinetic state is equilibrium: Pi_neq = 0 and g_i = g_i_eq.
A short relaxation transient is consequently possible.
This configuration represents low-Mach, isothermal, resistive MHD; it does
not evolve an energy equation or model the compressible shock benchmark
often also called the Orszag–Tang vortex.

## Equations and coupling

The target continuum equations, for constant ν and η, are:

```text
∂t rho + ∂α(rho uα) = 0
∂t(rho uα) + ∂β[rho uα uβ + rho cs² δαβ + Mαβ] = ∂β σ αβ
∂t Bα + ∂β(uβ Bα - Bβ uα) = η ∇² Bα
Mαβ = (|B|²/2) δαβ - Bα Bβ
∇·B = 0.
```

For a solenoidal field, -∂β Mαβ = [(∇×B)×B]α. The magnetic stress enters
the fluid equilibrium through

```text
f_i_eq,MHD = f_i_eq,R3 + w_i H_i^(2):M / (2 cs⁴).
```

This Hermite projection contributes zero mass and momentum and exactly M
to the second moment. Its trace subtraction matters in three dimensions.
There is no additional Guo force or half-force velocity shift in this case:
the momentum transfer is already represented by M.

The fluid update reconstructs

```text
f_i_star = f_i_eq,MHD + (1 - omega) f_i_neq,R3
Pi_neq = sum_i f_i c_i c_i - rho u u - rho cs² I - M.
```

The existing R3 recursive function is reused with this MHD-relative Pi_neq.
This is a low-Mach closure choice: a general third-order MHD
Chapman–Enskog derivation and finite-Mach accuracy study are still needed
before making higher-order accuracy claims.

For the magnetic lattice:

```text
W0 = 1/4, W1...W6 = 1/8, C² = 1/4
g_iα_eq = Wi [Bα + ξiβ (uβ Bα - Bβ uα)/C²]
g_iα_star = (1 - omegaB) g_iα + omegaB g_iα_eq
Bα = sum_i g_iα
tauB = 1/2 + η/C².
```

Each direction carries three magnetic components. The term -Bβ uα
produces magnetic stretching and is required by the induction equation.
Three independent passive-scalar equations would omit it.

## State and time stepping

| Persistent data per buffer | Scalars per stored node |
|---|---:|
| Fluid rho, ux, uy, uz, six Pi_neq components | 10 |
| Cached Bx, By, Bz | 3 |
| Seven vector magnetic populations | 21 |
| Total | 34 |

Two buffers use 272 bytes per stored node in float precision, before
128-byte field alignment. Ghost planes add overhead relative to the
physical grid, especially when Nz = 1. The exact allocation is printed
and saved in metadata. Hydrodynamic populations remain locally reconstructed;
the magnetic subsystem currently retains its populations. B alone cannot
retain the independent nonequilibrium kinetic information of a BGK solver.

One thread owns one destination node. It pulls collided magnetic populations
from buffer A, sums B at the destination, then pulls reconstructed fluid
populations from A. It subtracts M evaluated using the destination's new B
when recovering Pi_neq. All writes go to buffer B, followed by a whole-state
A/B swap. Periodic pull indices address physical nodes directly, including
when only one z plane exists.

## Output

Open `output/MHD/orszag_tang/fields.pvd` in ParaView. Its time values are t*.
All VTI arrays exclude ghosts and use lattice-unit coordinates.

- VTI: rho, velocity, B, current = curl B, divB, local kinetic energy
  rho |u|²/2, and local magnetic energy |B|²/2.
- CSV: mass sum, volume-averaged kinetic/magnetic/total energy, enstrophy,
  mean |J|², divergence RMS and maximum, relative divergence
  L RMS(div B)/RMS(B), maximum Mach number, density extrema, mean momentum,
  and mean magnetic flux.
- JSON: grid, nondimensionalization, transport parameters, allocation size,
  and diagnostic definitions.

Diagnostics use double-precision host accumulation. Spatial derivatives
are centered, periodic finite differences, with Δx = 1. They are not an
exact discrete solenoidal constraint. No divergence projection/cleaning
is implemented; monitor divergence under mesh refinement. At finite
divergence, the conservative stress coupling includes a B div B term
relative to J × B.

Sampling transfers seven fields to the CPU and writes ASCII VTI data.
This is a baseline for validation; GPU reductions and binary field output
would be appropriate before performance benchmarking. No GPU throughput
claim is made here.

## Verification

```bash
bash cases/MHD/tests/run_cpu.sh
MHD_TEST_NZ=3 bash cases/MHD/tests/run_cpu.sh
```

The tests compile the actual per-node numerical routines as C++20,
using a test-only allocation shim. They do not compile CUDA kernels or
emulate a GPU. On a 32 × 32 × 1 mesh at Re = Rm = 10 they check:

- Equilibrium mass, momentum, Maxwell stress, and magnetic flux moments
  for arbitrary three-component states.
- Uniform-state preservation with NaN ghost cells and independent
  magnetic kinetic-mode streaming checks, including multiple z planes.
- Exact reduction to the existing fluid update when B = 0.
- Force-free resistive decay against exp(-η k² t).
- Coupled Alfvén propagation and damping for ν = η.
- Orszag–Tang initial energies, finite-difference enstrophy/current,
  short evolution, mass/flux conservation, and divergence.
- Parseable VTI/PVD/CSV/JSON output with correct field sizes and energies.

Initial diagnostic references for a sufficiently resolved square mesh are:

```text
K(0) = rho0 U0²/2 - U0² B0²/(32 cs²)
M(0) = B0²/2
enstrophy_FD(0) = U0² sin²(k)/2
meanJ²_FD(0) = B0² [sin²(k) + sin²(2k)]/2
divB_FD(0) = 0.
```

CPU checks passed with both one and three physical z planes during implementation:
resistive-wave relative L2 error
0.256%, Alfvén-wave error 0.263%, and relative mass drift 7.09e-7 after
100 Orszag–Tang steps. The centered divergence RMS in that coarse test
was 3.65e-5 in lattice units.

CUDA compilation, GPU execution, and a full default-resolution benchmark
have not been performed in the implementation environment, which lacks
nvcc and a CUDA device. Run the GPU smoke command above, then the default
case, and check temporal evolution and mesh convergence before research use.

## References

- [Dellar (2002), Lattice kinetic schemes for magnetohydrodynamics](https://people.maths.ox.ac.uk/dellar/latticeMHD.html),
  Journal of Computational Physics 179, 95–126, DOI 10.1006/jcph.2002.7044.
- [Foldes et al., Efficient kinetic Lattice Boltzmann simulation of three-dimensional Hall-MHD Turbulence](https://arxiv.org/abs/2212.11024).
  The present magnetic equilibrium is the zero-Hall formulation. This
  implementation does not include Hall physics or their central-moment
  fluid collision model.

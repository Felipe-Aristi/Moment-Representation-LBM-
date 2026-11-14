
#ifndef JET_BOUNDARY_CONDITIONS_CUH
#define JET_BOUNDARY_CONDITIONS_CUH

#include "../../src/constants.cuh"
#include "../../src/memory.cuh"

#include "../../src/utilities/indexing.cuh"
#include "../../src/utilities/types.cuh"
#include "../../src/utilities/mathUtilities.cuh"

// ============================================================
// Jet inlet
// ============================================================

__device__ __forceinline__
void jetInletBC(const label_t x, const label_t z, const MomentState &B, const label_t completedStep) noexcept
{
    constexpr label_t yB = static_cast<label_t>(0);
    constexpr label_t yF = static_cast<label_t>(1);

    const label_t idB = idx(x, yB, z);
    const label_t idF = idx(x, yF, z);

    const real_t uyB = jetInletVelocity(x, z, completedStep);

    B.rho[idB] = rho0;

    B.ux[idB] = static_cast<real_t>(0);
    B.uy[idB] = uyB;
    B.uz[idB] = static_cast<real_t>(0);

    B.pixx[idB] = B.pixx[idF];
    B.pixy[idB] = B.pixy[idF];
    B.pixz[idB] = B.pixz[idF];

    B.piyy[idB] = B.piyy[idF];
    B.piyz[idB] = B.piyz[idF];
    B.pizz[idB] = B.pizz[idF];
}

// ============================================================
// Neumann outlet
//
// Zero normal gradient:
// phi(yB) = phi(yF)
//
// Applied to density, velocity and non-equilibrium moments.
// ============================================================

__device__ __forceinline__
void jetOutletBC(const label_t x, const label_t z, const MomentState &B) noexcept
{
    constexpr label_t yB = NY - static_cast<label_t>(1);
    constexpr label_t yF = NY - static_cast<label_t>(2);

    const label_t idB = idx(x, yB, z);
    const label_t idF = idx(x, yF, z);

    // Density
    B.rho[idB] = B.rho[idF];

    // Velocity
    B.ux[idB] = B.ux[idF];
    B.uy[idB] = B.uy[idF];
    B.uz[idB] = B.uz[idF];

    // Non-equilibrium stress
    B.pixx[idB] = B.pixx[idF];
    B.pixy[idB] = B.pixy[idF];
    B.pixz[idB] = B.pixz[idF];

    B.piyy[idB] = B.piyy[idF];
    B.piyz[idB] = B.piyz[idF];
    B.pizz[idB] = B.pizz[idF];
}

// ============================================================
// Complete jet boundary conditions
// ============================================================

__device__ __forceinline__
void jetBoundaryConditions(const label_t x,
                           const label_t z,
                           const MomentState &A,
                           const MomentState &B,
                           const label_t completedStep) noexcept
{
    (void)A;

    jetInletBC(x, z, B, completedStep);
    jetOutletBC(x, z, B);
}

#endif // JET_BOUNDARY_CONDITIONS_CUH

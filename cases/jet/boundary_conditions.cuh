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

__device__ __forceinline__ void jetInletBC(const label_t x, const label_t z, const MomentState &B) noexcept
{
    constexpr label_t yB = static_cast<label_t>(0);
    constexpr label_t yF = static_cast<label_t>(1);

    const label_t idB = idx(x, yB, z);
    const label_t idF = idx(x, yF, z);

    const real_t uyB = isJet(x, z) * jet_velocity;

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
// Convective-Neumann jet outlet
// A = state at t
// B = state at t + 1
// ============================================================

__device__ __forceinline__ void jetOutletBC(const label_t x, const label_t z,
                                            const MomentState &A, const MomentState &B) noexcept
{
    constexpr label_t yB = NY - static_cast<label_t>(1);
    constexpr label_t yF = NY - static_cast<label_t>(2);

    const label_t idB = idx(x, yB, z);
    const label_t idF = idx(x, yF, z);

    real_t uc = A.uy[idF];

    if (uc < static_cast<real_t>(0))
    {
        uc = static_cast<real_t>(0);
    }

    if (uc > static_cast<real_t>(1))
    {
        uc = static_cast<real_t>(1);
    }

    B.rho[idB] = convectiveOutlet(A.rho[idB], A.rho[idF], uc);

    B.ux[idB] = convectiveOutlet(A.ux[idB], A.ux[idF], uc);
    B.uy[idB] = convectiveOutlet(A.uy[idB], A.uy[idF], uc);
    B.uz[idB] = convectiveOutlet(A.uz[idB], A.uz[idF], uc);

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

__device__ __forceinline__ void jetBoundaryConditions(const label_t x, const label_t z,
                                                      const MomentState &A, const MomentState &B) noexcept
{
    jetInletBC(x, z, B);
    jetOutletBC(x, z, A, B);
}
#endif
#ifndef CHANNEL_BOUNDARY_CONDITIONS_CUH
#define CHANNEL_BOUNDARY_CONDITIONS_CUH

#include "../../src/constants.cuh"
#include "../../src/memory.cuh"

#include "../../src/utilities/indexing.cuh"
#include "../../src/utilities/types.cuh"

// ============================================================
// Lower no-slip wall
//
// A = state at t
// B = state at t + 1
// ============================================================

__device__ __forceinline__ void channelLowerWallBC(const label_t x, const label_t z,
                                                   const MomentState &A, const MomentState &B) noexcept
{
    constexpr label_t yB = static_cast<label_t>(0);
    constexpr label_t yF = static_cast<label_t>(1);

    const label_t idB = idx(x, yB, z);
    const label_t idF = idx(x, yF, z);

    B.rho[idB] = B.rho[idF];

    B.ux[idB] = static_cast<real_t>(0);
    B.uy[idB] = static_cast<real_t>(0);
    B.uz[idB] = static_cast<real_t>(0);

    B.pixx[idB] = B.pixx[idF];
    B.pixy[idB] = B.pixy[idF];
    B.pixz[idB] = B.pixz[idF];

    B.piyy[idB] = B.piyy[idF];
    B.piyz[idB] = B.piyz[idF];
    B.pizz[idB] = B.pizz[idF];
}

__device__ __forceinline__ void channelUpperWallBC(const label_t x, const label_t z,
                                                   const MomentState &A, const MomentState &B) noexcept
{
    constexpr label_t yB = NY - static_cast<label_t>(1);
    constexpr label_t yF = NY - static_cast<label_t>(2);

    const label_t idB = idx(x, yB, z);
    const label_t idF = idx(x, yF, z);

    B.rho[idB] = B.rho[idF];

    B.ux[idB] = static_cast<real_t>(0);
    B.uy[idB] = static_cast<real_t>(0);
    B.uz[idB] = static_cast<real_t>(0);

    B.pixx[idB] = B.pixx[idF];
    B.pixy[idB] = B.pixy[idF];
    B.pixz[idB] = B.pixz[idF];

    B.piyy[idB] = B.piyy[idF];
    B.piyz[idB] = B.piyz[idF];
    B.pizz[idB] = B.pizz[idF];
}

__device__ __forceinline__ void channelBoundaryConditions(const label_t x, const label_t z,
                                                          const MomentState &A, const MomentState &B) noexcept
{
    channelLowerWallBC(x, z, A, B);
    channelUpperWallBC(x, z, A, B);
}

#endif
#ifndef MATH_UTILITIES_CUH
#define MATH_UTILITIES_CUH

#include <cmath>
#include "types.cuh"
#include "../constants.cuh"

// =======================================================
// Jet mask
// =======================================================

__device__ __forceinline__ bool isJet(const label_t x, const label_t z) noexcept
{
    const real_t dx = static_cast<real_t>(x) - jet_x0;
    const real_t dz = static_cast<real_t>(z) - jet_z0;

    return (dx * dx + dz * dz) <= (jet_radius * jet_radius) ? 1 : 0;
}

// ============================================================
// Convective update
// phi_B^(n+1) = phi_B^n - uc (phi_B^n - phi_F^n)
// ============================================================

__device__ __forceinline__ real_t convectiveOutlet(const real_t phiB, const real_t phiF, const real_t uc) noexcept
{
    return phiB - uc * (phiB - phiF);
}

#endif
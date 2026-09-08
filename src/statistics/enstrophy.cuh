#ifndef ENSTROPHY_CUH
#define ENSTROPHY_CUH

#include <cstddef>

#include "../constants.cuh"
#include "../memory.cuh"

#include "../utilities/indexing.cuh"
#include "../utilities/types.cuh"

// ============================================================
// Physical coordinates
// ============================================================

__host__ __device__ constexpr __forceinline__ void physicalCoordinates(const std::size_t physicalId,
                                                                       label_t &x,
                                                                       label_t &y,
                                                                       label_t &z) noexcept
{
    const std::size_t xPhysical = physicalId % static_cast<std::size_t>(NXPhysical);
    const std::size_t yzPhysical = physicalId / static_cast<std::size_t>(NXPhysical);
    const std::size_t yPhysical = yzPhysical % static_cast<std::size_t>(NYPhysical);
    const std::size_t zPhysical = yzPhysical / static_cast<std::size_t>(NYPhysical);

    x = static_cast<label_t>(xPhysical) + static_cast<label_t>(1);
    y = static_cast<label_t>(yPhysical) + static_cast<label_t>(1);
    z = static_cast<label_t>(zPhysical) + static_cast<label_t>(1);
}

// ============================================================
// First periodic neighbors
// ============================================================

__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t physicalMinusOne(const label_t coordinate,
                                                                                     const label_t physicalExtent) noexcept
{
    return coordinate == static_cast<label_t>(1) ? physicalExtent : coordinate - static_cast<label_t>(1);
}

__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t physicalPlusOne(const label_t coordinate,
                                                                                    const label_t physicalExtent) noexcept
{
    return coordinate == physicalExtent ? static_cast<label_t>(1) : coordinate + static_cast<label_t>(1);
}

// ============================================================
// Second periodic neighbors
// ============================================================

__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t physicalMinusTwo(const label_t coordinate,
                                                                                     const label_t physicalExtent) noexcept
{
    return coordinate <= static_cast<label_t>(2)
               ? coordinate + physicalExtent - static_cast<label_t>(2)
               : coordinate - static_cast<label_t>(2);
}

__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t physicalPlusTwo(const label_t coordinate,
                                                                                    const label_t physicalExtent) noexcept
{
    return coordinate >= physicalExtent - static_cast<label_t>(1)
               ? coordinate - physicalExtent + static_cast<label_t>(2)
               : coordinate + static_cast<label_t>(2);
}

// ============================================================
// Fourth-order centered derivative
//
// df/dx = (-f_{i+2} + 8 f_{i+1} - 8 f_{i-1} + f_{i-2}) / 12
//
// dx = 1 in lattice units.
// ============================================================

__host__ __device__ [[nodiscard]] constexpr __forceinline__ statistics_t fourthOrderDerivative(const real_t fm2,
                                                                                               const real_t fm1,
                                                                                               const real_t fp1,
                                                                                               const real_t fp2) noexcept
{
    return (-static_cast<statistics_t>(fp2) + static_cast<statistics_t>(8) * static_cast<statistics_t>(fp1) - static_cast<statistics_t>(8) * static_cast<statistics_t>(fm1) + static_cast<statistics_t>(fm2)) / static_cast<statistics_t>(12);
}

// ============================================================
// Local enstrophy
//
// zeta_local = 0.5 * |curl(u)|^2
//
// Fourth-order centered derivatives evaluated directly on the
// physical periodic domain. Ghost nodes are never accessed.
// ============================================================

__device__ [[nodiscard]] __forceinline__ statistics_t enstrophy(const std::size_t physicalId,
                                                                const MomentState &state) noexcept
{
    label_t x, y, z;

    physicalCoordinates(physicalId, x, y, z);

    const label_t xm1 = physicalMinusOne(x, NXPhysical);
    const label_t xp1 = physicalPlusOne(x, NXPhysical);
    const label_t xm2 = physicalMinusTwo(x, NXPhysical);
    const label_t xp2 = physicalPlusTwo(x, NXPhysical);

    const label_t ym1 = physicalMinusOne(y, NYPhysical);
    const label_t yp1 = physicalPlusOne(y, NYPhysical);
    const label_t ym2 = physicalMinusTwo(y, NYPhysical);
    const label_t yp2 = physicalPlusTwo(y, NYPhysical);

    const label_t zm1 = physicalMinusOne(z, NZPhysical);
    const label_t zp1 = physicalPlusOne(z, NZPhysical);
    const label_t zm2 = physicalMinusTwo(z, NZPhysical);
    const label_t zp2 = physicalPlusTwo(z, NZPhysical);

    const label_t id_xm1 = idx(xm1, y, z);
    const label_t id_xp1 = idx(xp1, y, z);
    const label_t id_xm2 = idx(xm2, y, z);
    const label_t id_xp2 = idx(xp2, y, z);

    const label_t id_ym1 = idx(x, ym1, z);
    const label_t id_yp1 = idx(x, yp1, z);
    const label_t id_ym2 = idx(x, ym2, z);
    const label_t id_yp2 = idx(x, yp2, z);

    const label_t id_zm1 = idx(x, y, zm1);
    const label_t id_zp1 = idx(x, y, zp1);
    const label_t id_zm2 = idx(x, y, zm2);
    const label_t id_zp2 = idx(x, y, zp2);

    const statistics_t duzdy = fourthOrderDerivative(state.uz[id_ym2], state.uz[id_ym1], state.uz[id_yp1], state.uz[id_yp2]);
    const statistics_t duydz = fourthOrderDerivative(state.uy[id_zm2], state.uy[id_zm1], state.uy[id_zp1], state.uy[id_zp2]);

    const statistics_t duxdz = fourthOrderDerivative(state.ux[id_zm2], state.ux[id_zm1], state.ux[id_zp1], state.ux[id_zp2]);
    const statistics_t duzdx = fourthOrderDerivative(state.uz[id_xm2], state.uz[id_xm1], state.uz[id_xp1], state.uz[id_xp2]);

    const statistics_t duydx = fourthOrderDerivative(state.uy[id_xm2], state.uy[id_xm1], state.uy[id_xp1], state.uy[id_xp2]);
    const statistics_t duxdy = fourthOrderDerivative(state.ux[id_ym2], state.ux[id_ym1], state.ux[id_yp1], state.ux[id_yp2]);

    const statistics_t omegaX = duzdy - duydz;
    const statistics_t omegaY = duxdz - duzdx;
    const statistics_t omegaZ = duydx - duxdy;

    return static_cast<statistics_t>(0.5) * (omegaX * omegaX + omegaY * omegaY + omegaZ * omegaZ);
}

#endif
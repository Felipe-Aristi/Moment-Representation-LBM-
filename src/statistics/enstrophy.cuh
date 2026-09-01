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
// Periodic physical neighbors
// ============================================================

__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t physicalMinus(const label_t coordinate,
                                                                                  const label_t physicalExtent) noexcept
{
    return coordinate == static_cast<label_t>(1) ? physicalExtent : coordinate - static_cast<label_t>(1);
}

__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t physicalPlus(const label_t coordinate,
                                                                                 const label_t physicalExtent) noexcept
{
    return coordinate == physicalExtent ? static_cast<label_t>(1) : coordinate + static_cast<label_t>(1);
}

// ============================================================
// Local enstrophy
//
// zeta_local = 0.5 * |curl(u)|^2
//
// Second-order centered derivatives are evaluated directly over
// the physical periodic domain. Ghost nodes are never accessed.
// ============================================================

__device__ [[nodiscard]] __forceinline__ statistics_t enstrophy(const std::size_t physicalId,
                                                                const MomentState &state) noexcept
{
    label_t x, y, z;

    physicalCoordinates(physicalId, x, y, z);

    const label_t xm = physicalMinus(x, NXPhysical);
    const label_t xp = physicalPlus(x, NXPhysical);

    const label_t ym = physicalMinus(y, NYPhysical);
    const label_t yp = physicalPlus(y, NYPhysical);

    const label_t zm = physicalMinus(z, NZPhysical);
    const label_t zp = physicalPlus(z, NZPhysical);

    const label_t id_xm = idx(xm, y, z);
    const label_t id_xp = idx(xp, y, z);

    const label_t id_ym = idx(x, ym, z);
    const label_t id_yp = idx(x, yp, z);

    const label_t id_zm = idx(x, y, zm);
    const label_t id_zp = idx(x, y, zp);

    constexpr statistics_t half = static_cast<statistics_t>(0.5);

    const statistics_t duzdy = half * (static_cast<statistics_t>(state.uz[id_yp]) - static_cast<statistics_t>(state.uz[id_ym]));
    const statistics_t duydz = half * (static_cast<statistics_t>(state.uy[id_zp]) - static_cast<statistics_t>(state.uy[id_zm]));

    const statistics_t duxdz = half * (static_cast<statistics_t>(state.ux[id_zp]) - static_cast<statistics_t>(state.ux[id_zm]));
    const statistics_t duzdx = half * (static_cast<statistics_t>(state.uz[id_xp]) - static_cast<statistics_t>(state.uz[id_xm]));

    const statistics_t duydx = half * (static_cast<statistics_t>(state.uy[id_xp]) - static_cast<statistics_t>(state.uy[id_xm]));
    const statistics_t duxdy = half * (static_cast<statistics_t>(state.ux[id_yp]) - static_cast<statistics_t>(state.ux[id_ym]));

    const statistics_t omegaX = duzdy - duydz;
    const statistics_t omegaY = duxdz - duzdx;
    const statistics_t omegaZ = duydx - duxdy;

    return half * (omegaX * omegaX + omegaY * omegaY + omegaZ * omegaZ);
}

#endif
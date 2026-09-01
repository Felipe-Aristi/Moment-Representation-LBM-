#ifndef KINETIC_ENERGY_CUH
#define KINETIC_ENERGY_CUH

#include <cstddef>

#include "../constants.cuh"
#include "../utilities/indexing.cuh"
#include "../utilities/types.cuh"

// ============================================================
// Local kinetic energy
// ============================================================

__host__ __device__ [[nodiscard]] constexpr __forceinline__ statistics_t kineticEnergy(const real_t ux, const real_t uy, const real_t uz) noexcept
{
    const statistics_t uxStatistics = static_cast<statistics_t>(ux);
    const statistics_t uyStatistics = static_cast<statistics_t>(uy);
    const statistics_t uzStatistics = static_cast<statistics_t>(uz);

    return static_cast<statistics_t>(0.5) * (uxStatistics * uxStatistics + uyStatistics * uyStatistics + uzStatistics * uzStatistics);
}

// ============================================================
// Physical node -> stored node
// ============================================================

__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t physicalNodeIndex(const std::size_t physicalId) noexcept
{
    const std::size_t xPhysical = physicalId % static_cast<std::size_t>(NXPhysical);
    const std::size_t yzPhysical = physicalId / static_cast<std::size_t>(NXPhysical);
    const std::size_t yPhysical = yzPhysical % static_cast<std::size_t>(NYPhysical);
    const std::size_t zPhysical = yzPhysical / static_cast<std::size_t>(NYPhysical);

    const label_t x = static_cast<label_t>(xPhysical) + static_cast<label_t>(1);
    const label_t y = static_cast<label_t>(yPhysical) + static_cast<label_t>(1);
    const label_t z = static_cast<label_t>(zPhysical) + static_cast<label_t>(1);

    return idx(x, y, z);
}

static_assert(physicalNodeIndex(static_cast<std::size_t>(0)) == idx(static_cast<label_t>(1), static_cast<label_t>(1), static_cast<label_t>(1)));
static_assert(physicalNodeIndex(Nphysical - static_cast<std::size_t>(1)) == idx(NXPhysical, NYPhysical, NZPhysical));

#endif // KINETIC_ENERGY_CUH

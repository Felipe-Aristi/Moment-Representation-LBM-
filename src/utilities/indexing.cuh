#ifndef INDEXING_CUH
#define INDEXING_CUH

#include "../constants.cuh"
#include "types.cuh"

// ============================================================
// Linear indexing
// ============================================================

__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t idx(const label_t x, const label_t y, const label_t z) noexcept
{
    return x + y * NX + z * NXNY;
}

#endif
#ifndef STATISTICS_REDUCTION_CUH
#define STATISTICS_REDUCTION_CUH

#include <cuda_runtime.h>

#include "../utilities/types.cuh"

// ============================================================
// Statistics reduction configuration
// ============================================================

inline constexpr label_t statisticsWarpSize = static_cast<label_t>(32);
inline constexpr label_t statisticsBlockSize = static_cast<label_t>(256);
inline constexpr label_t statisticsNodesPerThread = static_cast<label_t>(2);
inline constexpr label_t statisticsWarpsPerBlock = statisticsBlockSize / statisticsWarpSize;

static_assert(statisticsBlockSize % statisticsWarpSize == static_cast<label_t>(0));

// ============================================================
// Warp reduction
// ============================================================

__device__ __forceinline__ statistics_t warpReduceSum(statistics_t value) noexcept
{
    constexpr unsigned int fullWarpMask = 0xffffffffu;

#pragma unroll
    for (int offset = static_cast<int>(statisticsWarpSize / static_cast<label_t>(2)); offset > 0; offset /= 2)
    {
        value += __shfl_down_sync(fullWarpMask, value, offset);
    }

    return value;
}

// ============================================================
// Block reduction
// ============================================================

__device__ __forceinline__ statistics_t blockReduceSum(statistics_t value) noexcept
{
    __shared__ statistics_t warpSums[statisticsWarpsPerBlock];

    const label_t lane = static_cast<label_t>(threadIdx.x) % statisticsWarpSize;
    const label_t warp = static_cast<label_t>(threadIdx.x) / statisticsWarpSize;

    value = warpReduceSum(value);

    if (lane == static_cast<label_t>(0))
    {
        warpSums[warp] = value;
    }

    __syncthreads();

    value = threadIdx.x < statisticsWarpsPerBlock ? warpSums[lane] : static_cast<statistics_t>(0);

    if (warp == static_cast<label_t>(0))
    {
        value = warpReduceSum(value);
    }

    return value;
}

#endif // STATISTICS_REDUCTION_CUH

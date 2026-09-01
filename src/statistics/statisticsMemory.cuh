#ifndef STATISTICS_MEMORY_CUH
#define STATISTICS_MEMORY_CUH

#include <cuda_runtime.h>
#include <cstddef>

#include "../constants.cuh"
#include "../utilities/cudaUtilities.cuh"
#include "../utilities/types.cuh"
#include "reduction.cuh"

// ============================================================
// Statistics memory sizes
// ============================================================

inline constexpr std::size_t statisticsNodesPerBlock = static_cast<std::size_t>(statisticsBlockSize) * static_cast<std::size_t>(statisticsNodesPerThread);
inline constexpr std::size_t statisticsPartialCount = (Nphysical + statisticsNodesPerBlock - static_cast<std::size_t>(1)) / statisticsNodesPerBlock;
inline constexpr std::size_t bytesStatisticsPartials = statisticsPartialCount * sizeof(statistics_t);

static_assert(Nphysical > static_cast<std::size_t>(0));

// ============================================================
// Statistics memory
// ============================================================

struct StatisticsMemory
{
    statistics_t *d_partials = nullptr;
    statistics_t *d_result = nullptr;

    statistics_t hostResult = static_cast<statistics_t>(0);
};

// ============================================================
// Statistics allocation
// ============================================================

inline void allocateStatisticsMemory(StatisticsMemory &memory)
{
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&memory.d_partials), bytesStatisticsPartials));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&memory.d_result), sizeof(statistics_t)));
}

// ============================================================
// Statistics deallocation
// ============================================================

inline void freeStatisticsMemory(StatisticsMemory &memory)
{
    if (memory.d_partials != nullptr)
    {
        CUDA_CHECK(cudaFree(memory.d_partials));
    }

    if (memory.d_result != nullptr)
    {
        CUDA_CHECK(cudaFree(memory.d_result));
    }

    memory = {};
}

#endif // STATISTICS_MEMORY_CUH

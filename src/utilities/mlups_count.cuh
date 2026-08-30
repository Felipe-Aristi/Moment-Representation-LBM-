#ifndef MLUPS_COUNT_CUH
#define MLUPS_COUNT_CUH

#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

#include "../constants.cuh"
#include "cudaUtilities.cuh"
#include "types.cuh"

inline constexpr label_t mlupsEvery = static_cast<label_t>(100);
inline constexpr std::size_t Nfluid = static_cast<std::size_t>(NX - static_cast<label_t>(2)) * static_cast<std::size_t>(NY - static_cast<label_t>(2)) * static_cast<std::size_t>(NZ - static_cast<label_t>(2));

// ============================================================
// MLUPS counter
// ============================================================

struct MlupsCounter
{
    cudaEvent_t start;
    cudaEvent_t stop;

    real_t totalTimeMs = static_cast<real_t>(0);
    label_t totalSteps = static_cast<label_t>(0);
};

// ============================================================
// Create CUDA events
// ============================================================

inline void initializeMlupsCounter(MlupsCounter &counter)
{
    CUDA_CHECK(cudaEventCreate(&counter.start));
    CUDA_CHECK(cudaEventCreate(&counter.stop));
}

// ============================================================
// Start measurement
// ============================================================

inline void startMlupsCounter(MlupsCounter &counter)
{
    CUDA_CHECK(cudaEventRecord(counter.start));
}

// ============================================================
// Stop measurement and print MLUPS
// ============================================================

inline void printMlups(MlupsCounter &counter, const label_t step)
{
    CUDA_CHECK(cudaEventRecord(counter.stop));
    CUDA_CHECK(cudaEventSynchronize(counter.stop));

    float elapsedMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, counter.start, counter.stop));

    counter.totalTimeMs += static_cast<real_t>(elapsedMs);
    counter.totalSteps += mlupsEvery;

    const real_t updates = static_cast<real_t>(Nfluid) * static_cast<real_t>(mlupsEvery);
    const real_t totalUpdates = static_cast<real_t>(Nfluid) * static_cast<real_t>(counter.totalSteps);

    const real_t mlups = updates / (static_cast<real_t>(elapsedMs) * static_cast<real_t>(1000));
    const real_t meanMlups = totalUpdates / (counter.totalTimeMs * static_cast<real_t>(1000));

    std::cout << std::setw(12) << step
              << std::setw(18) << std::fixed << std::setprecision(2) << mlups
              << std::setw(18) << std::fixed << std::setprecision(2) << meanMlups
              << "\n";
}

// ============================================================
// Print header
// ============================================================

inline void printMlupsHeader()
{
    std::cout << std::setw(12) << "Step"
              << std::setw(18) << "MLUPS"
              << std::setw(18) << "Mean MLUPS"
              << "\n";

    std::cout << std::string(48, '-') << "\n";
}

// ============================================================
// Destroy CUDA events
// ============================================================

inline void destroyMlupsCounter(MlupsCounter &counter)
{
    CUDA_CHECK(cudaEventDestroy(counter.start));
    CUDA_CHECK(cudaEventDestroy(counter.stop));
}

#endif // MLUPS_COUNT_CUH
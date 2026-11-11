
#ifndef CHANNEL_FLOW_STATISTICS_CUH
#define CHANNEL_FLOW_STATISTICS_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/constants.cuh"
#include "../../src/memory.cuh"
#include "../../src/utilities/cudaUtilities.cuh"
#include "../../src/utilities/indexing.cuh"
#include "../../src/utilities/types.cuh"

// ============================================================
// Channel statistics: ten FP32 values per wall-normal plane
//
// Field-major layout:
// index = moment * NY + y
//
// 0 <ux>    1 <uy>    2 <uz>
// 3 <ux^2>  4 <uy^2>  5 <uz^2>
// 6 <ux uy>
// 7 <ux^3>  8 <uy^3>  9 <uz^3>
//
// Averages are over physical x-z nodes and sampled times.
// No full-volume statistical fields are stored.
// ============================================================

inline constexpr int channelMomentFields = 10;
inline constexpr int channelMomentThreads = 256;
inline constexpr int channelMomentWarpSize = 32;
inline constexpr int channelMomentWarps = channelMomentThreads / channelMomentWarpSize;

inline constexpr std::size_t channelMomentValues =
    static_cast<std::size_t>(channelMomentFields) * NY;

inline constexpr std::size_t channelMomentBytes =
    channelMomentValues * sizeof(statistics_t);

static_assert(sizeof(statistics_t) == sizeof(float), "Channel statistics require FP32");
static_assert(channelMomentThreads % channelMomentWarpSize == 0);
static_assert(channelMomentWarps <= channelMomentWarpSize);
static_assert(NXPhysical > 0 && NZPhysical > 0 && NY > 2);

// ============================================================
// Channel-flow statistics memory
// ============================================================

struct ChannelFlowStatistics
{
    statistics_t *deviceAverages = nullptr;

    std::vector<statistics_t> hostAverages{};

    std::filesystem::path directory{};

    std::size_t sampleCount = 0;

    label_t firstStep = 0;
    label_t lastStep = 0;
};

// ============================================================
// Channel-flow moments kernel
//
// One CUDA block per wall-normal plane.
//
// 1. Each thread accumulates its x-z samples in registers.
// 2. Warp shuffles reduce the ten local quantities.
// 3. The first warp reduces the eight warp partial sums.
// 4. One thread updates each plane's running averages.
//
// No atomic operations are required.
// ============================================================

__global__ void channelFlowMomentsKernel(
    const MomentState state,
    statistics_t *averages,
    const statistics_t inverseSampleCount)
{
    const label_t y = static_cast<label_t>(blockIdx.x);

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (channelMomentWarpSize - 1);
    const int warp = tid / channelMomentWarpSize;

    // ========================================================
    // Local moment accumulators
    // ========================================================

    statistics_t local[channelMomentFields] = {};

    constexpr std::size_t planeSize =
        static_cast<std::size_t>(NXPhysical) * NZPhysical;

    constexpr statistics_t inversePlaneSize =
        static_cast<statistics_t>(1) /
        static_cast<statistics_t>(planeSize);

    // ========================================================
    // Accumulate local velocity moments
    //
    // y = 0 and y = NY - 1 are the two no-slip walls.
    //
    // Periodic ghost planes in x and z are excluded.
    // ========================================================

    if (y > static_cast<label_t>(0) &&
        y < NY - static_cast<label_t>(1))
    {
        for (std::size_t p = static_cast<std::size_t>(tid);
             p < planeSize;
             p += channelMomentThreads)
        {
            const label_t x =
                static_cast<label_t>(1) +
                static_cast<label_t>(p % NXPhysical);

            const label_t z =
                static_cast<label_t>(1) +
                static_cast<label_t>(p / NXPhysical);

            const label_t id = idx(x, y, z);

            // ====================================================
            // Instantaneous velocity
            // ====================================================

            const statistics_t ux = state.ux[id];
            const statistics_t uy = state.uy[id];
            const statistics_t uz = state.uz[id];

            // ====================================================
            // Second-order products
            // ====================================================

            const statistics_t ux2 = ux * ux;
            const statistics_t uy2 = uy * uy;
            const statistics_t uz2 = uz * uz;

            // ====================================================
            // First-order moments
            // ====================================================

            local[0] += ux;
            local[1] += uy;
            local[2] += uz;

            // ====================================================
            // Second-order moments
            // ====================================================

            local[3] += ux2;
            local[4] += uy2;
            local[5] += uz2;

            local[6] += ux * uy;

            // ====================================================
            // Third-order moments
            // ====================================================

            local[7] += ux2 * ux;
            local[8] += uy2 * uy;
            local[9] += uz2 * uz;
        }
    }

    // ========================================================
    // Stage 1: warp-level reductions
    //
    // All 32 threads in each warp participate.
    // ========================================================

    for (int offset = channelMomentWarpSize / 2;
         offset > 0;
         offset /= 2)
    {
        #pragma unroll
        for (int m = 0; m < channelMomentFields; ++m)
        {
            local[m] +=
                __shfl_down_sync(0xffffffffu, local[m], offset);
        }
    }

    // ========================================================
    // Stage 2: store one partial sum per warp
    //
    // Shared memory:
    // 10 moments * 8 warps * 4 bytes = 320 bytes
    // ========================================================

    __shared__ statistics_t
        warpSums[channelMomentFields][channelMomentWarps];

    if (lane == 0)
    {
        #pragma unroll
        for (int m = 0; m < channelMomentFields; ++m)
        {
            warpSums[m][warp] = local[m];
        }
    }

    __syncthreads();

    // ========================================================
    // Stage 3: final reduction by the first warp
    // ========================================================

    if (warp == 0)
    {
        #pragma unroll
        for (int m = 0; m < channelMomentFields; ++m)
        {
            statistics_t sum =
                lane < channelMomentWarps
                    ? warpSums[m][lane]
                    : static_cast<statistics_t>(0);

            for (int offset = channelMomentWarpSize / 2;
                 offset > 0;
                 offset /= 2)
            {
                sum += __shfl_down_sync(
                    0xffffffffu, sum, offset
                );
            }

            // ====================================================
            // Update running average
            // ====================================================

            if (lane == 0)
            {
                const std::size_t pos =
                    static_cast<std::size_t>(m) * NY + y;

                const statistics_t planeMean =
                    sum * inversePlaneSize;

                averages[pos] +=
                    (planeMean - averages[pos]) *
                    inverseSampleCount;
            }
        }
    }
}

// ============================================================
// Initialize channel-flow statistics
// ============================================================

inline void initializeChannelFlowStatistics(
    ChannelFlowStatistics &stats)
{
    if (stats.deviceAverages != nullptr)
    {
        throw std::runtime_error(
            "Channel statistics already initialized"
        );
    }

    // ========================================================
    // Output directory
    // ========================================================

    stats.directory =
        std::filesystem::path("post") /
        "channelflow" /
        "validation" /
        "data" /
        (
            "RETAU" +
            std::to_string(static_cast<label_t>(channel_ReTau)) +
            "_" +
            std::to_string(NXPhysical) + "x" +
            std::to_string(NY) + "x" +
            std::to_string(NZPhysical)
        );

    // ========================================================
    // Initialize statistical counters
    // ========================================================

    stats.hostAverages.assign(
        channelMomentValues,
        static_cast<statistics_t>(0)
    );

    stats.sampleCount = 0;
    stats.firstStep = 0;
    stats.lastStep = 0;

    // ========================================================
    // Allocate GPU statistical memory
    // ========================================================

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&stats.deviceAverages),
        channelMomentBytes
    ));

    CUDA_CHECK(cudaMemset(
        stats.deviceAverages,
        0,
        channelMomentBytes
    ));
}

// ============================================================
// Launch channel-flow statistics
//
// Called after the A-B swap and only at scheduled steps.
// ============================================================

inline void launchChannelFlowStatistics(
    const MomentState &state,
    ChannelFlowStatistics &stats,
    const label_t completedStep)
{
    if (stats.deviceAverages == nullptr)
    {
        throw std::runtime_error(
            "Channel statistics not initialized"
        );
    }

    if (stats.sampleCount > 0 &&
        completedStep <= stats.lastStep)
    {
        throw std::runtime_error(
            "Channel statistics steps must increase strictly"
        );
    }

    if (stats.sampleCount ==
        std::numeric_limits<std::size_t>::max())
    {
        throw std::runtime_error(
            "Channel statistics sample count overflow"
        );
    }

    // ========================================================
    // Incremental time-average weight
    // ========================================================

    const std::size_t nextCount = stats.sampleCount + 1;

    const statistics_t inverseSampleCount =
        static_cast<statistics_t>(1) /
        static_cast<statistics_t>(nextCount);

    // ========================================================
    // Launch one block per y plane
    // ========================================================

    channelFlowMomentsKernel
        <<<static_cast<unsigned int>(NY), channelMomentThreads>>>(
            state,
            stats.deviceAverages,
            inverseSampleCount
        );

    CUDA_CHECK(cudaGetLastError());

    // ========================================================
    // Sample metadata
    // ========================================================

    if (stats.sampleCount == 0)
    {
        stats.firstStep = completedStep;
    }

    stats.sampleCount = nextCount;
    stats.lastStep = completedStep;
}

// ============================================================
// Write channel-flow statistics
//
// Only 10 * NY floats are copied from GPU to CPU.
// No full 3D velocity fields are transferred.
// ============================================================

inline void writeChannelFlowStatistics(
    ChannelFlowStatistics &stats)
{
    if (stats.sampleCount == 0)
    {
        return;
    }

    std::filesystem::create_directories(stats.directory);

    // ========================================================
    // Copy statistical moments to host
    // ========================================================

    CUDA_CHECK(cudaMemcpy(
        stats.hostAverages.data(),
        stats.deviceAverages,
        channelMomentBytes,
        cudaMemcpyDeviceToHost
    ));

    // ========================================================
    // Write binary moments
    // ========================================================

    std::ofstream binary(
        stats.directory / "channel_moments.bin",
        std::ios::out |
        std::ios::binary |
        std::ios::trunc
    );

    if (!binary)
    {
        throw std::runtime_error(
            "Failed opening channel_moments.bin"
        );
    }

    binary.write(
        reinterpret_cast<const char *>(
            stats.hostAverages.data()
        ),
        static_cast<std::streamsize>(channelMomentBytes)
    );

    if (!binary)
    {
        throw std::runtime_error(
            "Failed writing channel_moments.bin"
        );
    }

    binary.close();

    if (!binary)
    {
        throw std::runtime_error(
            "Failed closing channel_moments.bin"
        );
    }

    // ========================================================
    // Write metadata
    // ========================================================

    std::ofstream meta(
        stats.directory / "channel_statistics_metadata.txt",
        std::ios::out |
        std::ios::trunc
    );

    if (!meta)
    {
        throw std::runtime_error(
            "Failed opening channel_statistics_metadata.txt"
        );
    }

    meta << std::setprecision(
                std::numeric_limits<real_t>::max_digits10
            )
         << "NXPhysical " << NXPhysical << '\n'
         << "NY " << NY << '\n'
         << "NZPhysical " << NZPhysical << '\n'
         << "ReTau " << channel_ReTau << '\n'
         << "halfHeight " << channel_halfHeight << '\n'
         << "uTau " << channel_uTau << '\n'
         << "nu " << channel_nu << '\n'
         << "g " << channel_g << '\n'
         << "sample_count " << stats.sampleCount << '\n'
         << "first_step " << stats.firstStep << '\n'
         << "last_step " << stats.lastStep << '\n'
         << "moment_fields 10\n"
         << "moment_names ux uy uz ux2 uy2 uz2 uxuy ux3 uy3 uz3\n"
         << "moment_dtype float32\n"
         << "moment_layout field_major_field_times_NY_plus_y\n"
         << "wall_normal y\n";

    if (!meta)
    {
        throw std::runtime_error(
            "Failed writing channel statistics metadata"
        );
    }
}

// ============================================================
// Free channel-flow statistical memory
// ============================================================

inline void freeChannelFlowStatistics(
    ChannelFlowStatistics &stats)
{
    if (stats.deviceAverages != nullptr)
    {
        CUDA_CHECK(cudaFree(stats.deviceAverages));
    }

    stats = {};
}

#endif // CHANNEL_FLOW_STATISTICS_CUH

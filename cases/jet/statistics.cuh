
#ifndef JET_STATISTICS_MR_CUH
#define JET_STATISTICS_MR_CUH

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

#include "../../src/constants.cuh"
#include "../../src/memory.cuh"
#include "../../src/utilities/indexing.cuh"
#include "../../src/utilities/cudaConfig.cuh"
#include "../../src/utilities/cudaUtilities.cuh"

// ============================================================
// Jet statistics precision
// ============================================================

using profile_stat_t = double;
using profile_count_t = unsigned long long;

// ============================================================
// Radial statistics dimensions
// ============================================================

inline constexpr label_t jetStatsBins =
    (NXPhysical < NZPhysical ? NXPhysical : NZPhysical) / static_cast<label_t>(2);

inline constexpr std::size_t jetStatsCells =
    static_cast<std::size_t>(NYPhysical) * static_cast<std::size_t>(jetStatsBins);

inline constexpr std::size_t jetStatsBytes =
    jetStatsCells * sizeof(profile_stat_t);

inline constexpr std::size_t jetCountBytes =
    jetStatsCells * sizeof(profile_count_t);

static_assert(jetStatsBins >= static_cast<label_t>(2));
static_assert(jetStatsSampleEvery > static_cast<label_t>(0));

// ============================================================
// GPU statistics memory
// ============================================================

struct JetRadialDevice
{
    profile_stat_t *sum_uy = nullptr;
    profile_stat_t *sum_uy2 = nullptr;
    profile_stat_t *sum_ur = nullptr;
    profile_stat_t *sum_ur2 = nullptr;
    profile_stat_t *sum_uruy = nullptr;

    profile_count_t *count = nullptr;
};

// ============================================================
// Runtime statistics
// ============================================================

struct JetRadialStatistics
{
    JetRadialDevice d{};
    std::filesystem::path directory{};

    unsigned long long sampledSteps = 0;

    label_t firstSampleStep = 0;
    label_t lastSampleStep = 0;
};

// ============================================================
// Radial statistics accumulation
// ============================================================

__global__ void accumulateJetRadialStatisticsKernel(
    const MomentState s,
    JetRadialDevice sums)
{
    const label_t x = static_cast<label_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const label_t y = static_cast<label_t>(blockIdx.y * blockDim.y + threadIdx.y);
    const label_t z = static_cast<label_t>(blockIdx.z * blockDim.z + threadIdx.z);

    if (x == 0 || y == 0 || z == 0 || x >= NX - 1 || y >= NY - 1 || z >= NZ - 1)
        return;

    const float dx = static_cast<float>(x) - static_cast<float>(jet_x0);
    const float dz = static_cast<float>(z) - static_cast<float>(jet_z0);

    const float r = sqrt(dx * dx + dz * dz);

    const label_t radialBin = static_cast<label_t>(r);

    if (radialBin >= jetStatsBins)
        return;

    const std::size_t cell =
        static_cast<std::size_t>(y - 1) * static_cast<std::size_t>(jetStatsBins) + radialBin;

    const label_t id = idx(x, y, z);

    const float vx = static_cast<float>(s.ux[id]);
    const float vy = static_cast<float>(s.uy[id]);
    const float vz = static_cast<float>(s.uz[id]);

    const float ur = r > 0.0f ? (dx * vx + dz * vz) / r : 0.0f;

    // Convert before multiplication to accumulate all
    // radial velocity moments in double precision.

    const profile_stat_t vyStat = static_cast<profile_stat_t>(vy);
    const profile_stat_t urStat = static_cast<profile_stat_t>(ur);

    atomicAdd(sums.sum_uy + cell, vyStat);
    atomicAdd(sums.sum_uy2 + cell, vyStat * vyStat);

    atomicAdd(sums.sum_ur + cell, urStat);
    atomicAdd(sums.sum_ur2 + cell, urStat * urStat);

    atomicAdd(sums.sum_uruy + cell, urStat * vyStat);

    atomicAdd(sums.count + cell, static_cast<profile_count_t>(1));
}

// ============================================================
// Initialize radial statistics
// ============================================================

inline void initializeJetRadialStatistics(JetRadialStatistics &s)
{
    s.directory =
        std::filesystem::path("post") / "jet" / "validation" / "data" /
        ("Re" + std::to_string(static_cast<label_t>(jet_Re))) / "mean_profiles";

    std::filesystem::create_directories(s.directory);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&s.d.sum_uy), jetStatsBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&s.d.sum_uy2), jetStatsBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&s.d.sum_ur), jetStatsBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&s.d.sum_ur2), jetStatsBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&s.d.sum_uruy), jetStatsBytes));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&s.d.count), jetCountBytes));

    CUDA_CHECK(cudaMemset(s.d.sum_uy, 0, jetStatsBytes));
    CUDA_CHECK(cudaMemset(s.d.sum_uy2, 0, jetStatsBytes));
    CUDA_CHECK(cudaMemset(s.d.sum_ur, 0, jetStatsBytes));
    CUDA_CHECK(cudaMemset(s.d.sum_ur2, 0, jetStatsBytes));
    CUDA_CHECK(cudaMemset(s.d.sum_uruy, 0, jetStatsBytes));

    CUDA_CHECK(cudaMemset(s.d.count, 0, jetCountBytes));

    s.sampledSteps = 0;
    s.firstSampleStep = 0;
    s.lastSampleStep = 0;
}

// ============================================================
// Statistical sampling criterion
// ============================================================

[[nodiscard]] inline bool jetRadialSampleDue(const label_t completedStep) noexcept
{
    const float tstar =
        static_cast<float>(completedStep) *
        static_cast<float>(jet_velocity) /
        static_cast<float>(jet_diameter);

    return completedStep % jetStatsSampleEvery == 0 &&
           tstar >= static_cast<float>(jetStatsStartTStar);
}

// ============================================================
// Launch GPU statistics kernel
// ============================================================

inline void launchJetRadialStatistics(
    const CudaConfig &cfg,
    const MomentState &fluid,
    JetRadialStatistics &s,
    const label_t completedStep)
{
    if (!jetRadialSampleDue(completedStep))
        return;

    accumulateJetRadialStatisticsKernel<<<cfg.grid, cfg.block>>>(fluid, s.d);

    CUDA_CHECK(cudaGetLastError());

    if (s.sampledSteps == 0)
        s.firstSampleStep = completedStep;

    s.lastSampleStep = completedStep;
    ++s.sampledSteps;
}

// ============================================================
// Write binary arrays
// ============================================================

template <typename T>
inline void writeJetRadialArray(
    const std::filesystem::path &path,
    const T *data)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);

    if (!out)
        throw std::runtime_error("Cannot open " + path.string());

    out.write(
        reinterpret_cast<const char *>(data),
        static_cast<std::streamsize>(jetStatsCells * sizeof(T)));

    if (!out)
        throw std::runtime_error("Cannot write " + path.string());
}

// ============================================================
// Write radial statistics
// ============================================================

inline void writeJetRadialStatistics(const JetRadialStatistics &s)
{
    // FP64 host arrays

    std::vector<profile_stat_t> uy(jetStatsCells);
    std::vector<profile_stat_t> uy2(jetStatsCells);
    std::vector<profile_stat_t> ur(jetStatsCells);
    std::vector<profile_stat_t> ur2(jetStatsCells);
    std::vector<profile_stat_t> uruy(jetStatsCells);

    std::vector<profile_count_t> count(jetStatsCells);

    CUDA_CHECK(cudaMemcpy(uy.data(), s.d.sum_uy, jetStatsBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uy2.data(), s.d.sum_uy2, jetStatsBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ur.data(), s.d.sum_ur, jetStatsBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ur2.data(), s.d.sum_ur2, jetStatsBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uruy.data(), s.d.sum_uruy, jetStatsBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(count.data(), s.d.count, jetCountBytes, cudaMemcpyDeviceToHost));

    writeJetRadialArray(s.directory / "radial_sum_uy.bin", uy.data());
    writeJetRadialArray(s.directory / "radial_sum_uy2.bin", uy2.data());
    writeJetRadialArray(s.directory / "radial_sum_ur.bin", ur.data());
    writeJetRadialArray(s.directory / "radial_sum_ur2.bin", ur2.data());
    writeJetRadialArray(s.directory / "radial_sum_uruy.bin", uruy.data());

    writeJetRadialArray(s.directory / "radial_count.bin", count.data());

    // ========================================================
    // Metadata
    // ========================================================

    std::ofstream meta(s.directory / "radial_profile_metadata.txt", std::ios::trunc);

    if (!meta)
        throw std::runtime_error("Cannot write radial profile metadata");

    meta << std::setprecision(17);

    meta << "NX " << NXPhysical << '\n';
    meta << "NY " << NYPhysical << '\n';
    meta << "NZ " << NZPhysical << '\n';

    meta << "storedNX " << NX << '\n';
    meta << "storedNY " << NY << '\n';
    meta << "storedNZ " << NZ << '\n';

    meta << "NR_BINS " << jetStatsBins << '\n';
    meta << "NradialProfileCells " << jetStatsCells << '\n';

    meta << "D " << jet_diameter << '\n';
    meta << "jet_radius " << jet_radius << '\n';
    meta << "jet_velocity " << jet_velocity << '\n';
    meta << "Re " << jet_Re << '\n';

    meta << "stats_start_tstar " << jetStatsStartTStar << '\n';
    meta << "NSTATS_SAMPLE " << jetStatsSampleEvery << '\n';

    meta << "start_step " << s.firstSampleStep << '\n';
    meta << "final_sample_step " << s.lastSampleStep << '\n';
    meta << "radial_sample_count " << s.sampledSteps << '\n';

    meta << "layout y_major_index_equals_y_times_NR_BINS_plus_rbin\n";
    meta << "coordinate y_equals_stored_y_minus_1\n";
    meta << "radial_bins floor_radius_unit_width\n";

    meta << "stat_dtype float64\n";
    meta << "count_dtype uint64\n";

    if (!meta)
        throw std::runtime_error("Failed writing radial profile metadata");
}

// ============================================================
// Release statistics memory
// ============================================================

inline void freeJetRadialStatistics(JetRadialStatistics &s)
{
    if (s.d.sum_uy != nullptr)
        CUDA_CHECK(cudaFree(s.d.sum_uy));

    if (s.d.sum_uy2 != nullptr)
        CUDA_CHECK(cudaFree(s.d.sum_uy2));

    if (s.d.sum_ur != nullptr)
        CUDA_CHECK(cudaFree(s.d.sum_ur));

    if (s.d.sum_ur2 != nullptr)
        CUDA_CHECK(cudaFree(s.d.sum_ur2));

    if (s.d.sum_uruy != nullptr)
        CUDA_CHECK(cudaFree(s.d.sum_uruy));

    if (s.d.count != nullptr)
        CUDA_CHECK(cudaFree(s.d.count));

    s = JetRadialStatistics{};
}

#endif // JET_STATISTICS_MR_CUH

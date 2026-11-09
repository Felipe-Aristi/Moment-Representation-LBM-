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

// Same radial-bin width (one lattice spacing) as the old jet solver.
inline constexpr label_t jetStatsBins = (NXPhysical < NZPhysical ? NXPhysical : NZPhysical) / static_cast<label_t>(2);
inline constexpr std::size_t jetStatsCells = static_cast<std::size_t>(NYPhysical) * static_cast<std::size_t>(jetStatsBins);
inline constexpr std::size_t jetStatsBytes = jetStatsCells * sizeof(float);
inline constexpr std::size_t jetCountBytes = jetStatsCells * sizeof(unsigned long long);
static_assert(jetStatsBins >= static_cast<label_t>(2));
static_assert(jetStatsSampleEvery > static_cast<label_t>(0));

struct JetRadialDevice
{
    float *sum_uy = nullptr;
    float *sum_uy2 = nullptr;
    float *sum_ur = nullptr;
    float *sum_ur2 = nullptr;
    float *sum_uruy = nullptr;
    unsigned long long *count = nullptr;
};

struct JetRadialStatistics
{
    JetRadialDevice d{};
    std::filesystem::path directory{};
    unsigned long long sampledSteps = 0;
    label_t firstSampleStep = 0;
    label_t lastSampleStep = 0;
};

// Each cell is assigned uniquely to a CUDA thread, then accumulated
// into an axial-index/radial-bin position via atomic operations.
__global__ void accumulateJetRadialStatisticsKernel(const MomentState s, JetRadialDevice sums)
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

    const std::size_t cell = static_cast<std::size_t>(y - 1) * static_cast<std::size_t>(jetStatsBins) + radialBin;
    const label_t id = idx(x, y, z);

    const float vx = static_cast<float>(s.ux[id]);
    const float vy = static_cast<float>(s.uy[id]);
    const float vz = static_cast<float>(s.uz[id]);
    const float ur = r > 0.0 ? (dx * vx + dz * vz) / r : 0.0;

    atomicAdd(sums.sum_uy + cell, vy);
    atomicAdd(sums.sum_uy2 + cell, vy * vy);
    atomicAdd(sums.sum_ur + cell, ur);
    atomicAdd(sums.sum_ur2 + cell, ur * ur);
    atomicAdd(sums.sum_uruy + cell, ur * vy);
    atomicAdd(sums.count + cell, 1ULL);
}

inline void initializeJetRadialStatistics(JetRadialStatistics &s)
{
    s.directory = std::filesystem::path("post") / "jet" / "validation" / "data" /
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

[[nodiscard]] inline bool jetRadialSampleDue(const label_t completedStep) noexcept
{
    const float tstar = static_cast<float>(completedStep) * static_cast<float>(jet_velocity) /
                         static_cast<float>(jet_diameter);
    return completedStep % jetStatsSampleEvery == 0 && tstar >= static_cast<float>(jetStatsStartTStar);
}

inline void launchJetRadialStatistics(const CudaConfig &cfg, const MomentState &fluid,
                                      JetRadialStatistics &s, const label_t completedStep)
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

template <typename T>
inline void writeJetRadialArray(const std::filesystem::path &path, const T *data)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        throw std::runtime_error("Cannot open " + path.string());
    out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(jetStatsCells * sizeof(T)));
    if (!out)
        throw std::runtime_error("Cannot write " + path.string());
}

inline void writeJetRadialStatistics(const JetRadialStatistics &s)
{
    // Called rarely; small GPU -> host copies; no volumetric data transferred.
    std::vector<float> uy(jetStatsCells), uy2(jetStatsCells), ur(jetStatsCells);
    std::vector<float> ur2(jetStatsCells), uruy(jetStatsCells);
    std::vector<unsigned long long> count(jetStatsCells);
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

    std::ofstream meta(s.directory / "radial_profile_metadata.txt", std::ios::trunc);
    if (!meta)
        throw std::runtime_error("Cannot write radial profile metadata");
    meta << std::setprecision(17);
    // Physical dimensions (ghost nodes excluded), matching stored arrays.
    meta << "NX " << NXPhysical << '\n' << "NY " << NYPhysical << '\n' << "NZ " << NZPhysical << '\n';
    meta << "storedNX " << NX << '\n' << "storedNY " << NY << '\n' << "storedNZ " << NZ << '\n';
    meta << "NR_BINS " << jetStatsBins << '\n';
    meta << "NradialProfileCells " << jetStatsCells << '\n';
    meta << "D " << jet_diameter << '\n' << "jet_radius " << jet_radius << '\n';
    meta << "jet_velocity " << jet_velocity << '\n' << "Re " << jet_Re << '\n';
    meta << "stats_start_tstar " << jetStatsStartTStar << '\n';
    meta << "NSTATS_SAMPLE " << jetStatsSampleEvery << '\n';
    meta << "start_step " << s.firstSampleStep << '\n' << "final_sample_step " << s.lastSampleStep << '\n';
    meta << "radial_sample_count " << s.sampledSteps << '\n';
    meta << "layout y_major_index_equals_y_times_NR_BINS_plus_rbin\n";
    meta << "coordinate y_equals_stored_y_minus_1\n";
    meta << "radial_bins floor_radius_unit_width\n";
    meta << "stat_dtype float64\ncount_dtype uint64\n";
    if (!meta)
        throw std::runtime_error("Failed writing radial profile metadata");
}

inline void freeJetRadialStatistics(JetRadialStatistics &s)
{
    if (s.d.sum_uy != nullptr) CUDA_CHECK(cudaFree(s.d.sum_uy));
    if (s.d.sum_uy2 != nullptr) CUDA_CHECK(cudaFree(s.d.sum_uy2));
    if (s.d.sum_ur != nullptr) CUDA_CHECK(cudaFree(s.d.sum_ur));
    if (s.d.sum_ur2 != nullptr) CUDA_CHECK(cudaFree(s.d.sum_ur2));
    if (s.d.sum_uruy != nullptr) CUDA_CHECK(cudaFree(s.d.sum_uruy));
    if (s.d.count != nullptr) CUDA_CHECK(cudaFree(s.d.count));
    s = JetRadialStatistics{};
}

#endif


#ifndef JET_ENERGY_STATISTICS_CUH
#define JET_ENERGY_STATISTICS_CUH

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>

#include "../../src/constants.cuh"
#include "../../src/utilities/types.cuh"

// ============================================================
// Jet kinetic-energy history
// ============================================================

struct JetKineticEnergyHistory
{
    std::filesystem::path directory{};

    std::ofstream csv{};
    std::ofstream totalBin{};
    std::ofstream averageBin{};
    std::ofstream differenceBin{};

    statistics_t runningMean = static_cast<statistics_t>(0);
    statistics_t previousEnergy = static_cast<statistics_t>(0);

    std::size_t sampleCount = 0;
    label_t previousStep = static_cast<label_t>(0);
};

// ============================================================
// Initialize output
// ============================================================

inline void initializeJetKineticEnergyHistory(JetKineticEnergyHistory &h)
{
    static_assert(sizeof(statistics_t) == sizeof(float),
                  "Jet energy binary files require FP32 statistics_t");

    h.runningMean = static_cast<statistics_t>(0);
    h.previousEnergy = static_cast<statistics_t>(0);
    h.sampleCount = 0;
    h.previousStep = static_cast<label_t>(0);

    h.directory = std::filesystem::path("post") / "jet" / "validation" / "data" /
                  ("Re" + std::to_string(static_cast<label_t>(jet_Re))) / "mean_profiles";

    std::filesystem::create_directories(h.directory);

    h.csv.open(h.directory / "energy_history.csv", std::ios::out | std::ios::trunc);
    h.totalBin.open(h.directory / "tke_total.bin", std::ios::out | std::ios::binary | std::ios::trunc);
    h.averageBin.open(h.directory / "tke_avg.bin", std::ios::out | std::ios::binary | std::ios::trunc);
    h.differenceBin.open(h.directory / "tke_diff.bin", std::ios::out | std::ios::binary | std::ios::trunc);

    if (!h.csv || !h.totalBin || !h.averageBin || !h.differenceBin)
    {
        throw std::runtime_error("Failed opening jet kinetic-energy files in " + h.directory.string());
    }

    h.csv << "step,t_star,energy_total,energy_running_mean,relative_change,energy_star,energy_running_mean_star\n";
    h.csv << std::setprecision(std::numeric_limits<statistics_t>::max_digits10);

    if (!h.csv)
    {
        throw std::runtime_error("Failed writing jet kinetic-energy CSV header");
    }
}

// ============================================================
// Binary output - FP32
// ============================================================

inline void appendJetEnergyBinary(std::ofstream &out, const statistics_t value)
{
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));

    if (!out)
    {
        throw std::runtime_error("Failed writing jet kinetic-energy binary history");
    }
}

// ============================================================
// Record kinetic energy
// ============================================================

inline void recordJetKineticEnergy(JetKineticEnergyHistory &h, const label_t completedStep,
                                   const statistics_t energySum)
{
    if (!std::isfinite(energySum) || energySum < static_cast<statistics_t>(0))
    {
        throw std::runtime_error("Non-finite or negative domain kinetic energy");
    }

    // ========================================================
    // Check sample ordering
    // ========================================================

    if (h.sampleCount > 0 && completedStep <= h.previousStep)
    {
        throw std::runtime_error("Jet kinetic-energy steps must increase strictly");
    }

    // ========================================================
    // Energy normalization
    // ========================================================

    const statistics_t normalization = static_cast<statistics_t>(0.5) * jet_velocity * jet_velocity *
                                       static_cast<statistics_t>(Nphysical);

    if (!std::isfinite(normalization) || normalization <= static_cast<statistics_t>(0) ||
        jet_diameter <= static_cast<real_t>(0))
    {
        throw std::runtime_error("Invalid jet kinetic-energy normalization or diameter");
    }

    // ========================================================
    // Relative energy variation
    // ========================================================

    statistics_t relativeChange = std::numeric_limits<statistics_t>::quiet_NaN();

    if (h.sampleCount > 0 && h.previousEnergy > static_cast<statistics_t>(0))
    {
        relativeChange = std::abs(energySum - h.previousEnergy) / h.previousEnergy;
    }

    // ========================================================
    // Running average - FP32
    // ========================================================

    ++h.sampleCount;

    h.runningMean += (energySum - h.runningMean) / static_cast<statistics_t>(h.sampleCount);

    h.previousEnergy = energySum;
    h.previousStep = completedStep;

    // ========================================================
    // Dimensionless time and energy
    // ========================================================

    const statistics_t tStar = static_cast<statistics_t>(completedStep) * jet_velocity / jet_diameter;

    const statistics_t energyStar = energySum / normalization;
    const statistics_t runningMeanStar = h.runningMean / normalization;

    // ========================================================
    // CSV output
    // ========================================================

    h.csv << completedStep << ','
          << tStar << ','
          << energySum << ','
          << h.runningMean << ','
          << relativeChange << ','
          << energyStar << ','
          << runningMeanStar << '\n';

    if (!h.csv)
    {
        throw std::runtime_error("Failed writing jet kinetic-energy CSV record");
    }

    // ========================================================
    // Binary output
    // ========================================================

    appendJetEnergyBinary(h.totalBin, energySum);
    appendJetEnergyBinary(h.averageBin, h.runningMean);
    appendJetEnergyBinary(h.differenceBin, relativeChange);

    // ========================================================
    // Flush outputs for live post-processing
    // ========================================================

    h.csv.flush();
    h.totalBin.flush();
    h.averageBin.flush();
    h.differenceBin.flush();

    if (!h.csv || !h.totalBin || !h.averageBin || !h.differenceBin)
    {
        throw std::runtime_error("Failed flushing jet kinetic-energy history");
    }
}

// ============================================================
// Close output
// ============================================================

inline void closeJetKineticEnergyHistory(JetKineticEnergyHistory &h)
{
    h.csv.close();
    h.totalBin.close();
    h.averageBin.close();
    h.differenceBin.close();

    if (h.csv.fail() || h.totalBin.fail() || h.averageBin.fail() || h.differenceBin.fail())
    {
        throw std::runtime_error("Failed closing jet kinetic-energy history");
    }
}

#endif

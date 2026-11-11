
#ifndef CHANNEL_ENERGY_STATISTICS_CUH
#define CHANNEL_ENERGY_STATISTICS_CUH

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

#include "../../src/constants.cuh"
#include "../../src/utilities/types.cuh"

// ============================================================
// Channel kinetic-energy history
//
// The GPU reduction is already implemented in the solver.
// This helper stores only a small CSV time history.
// ============================================================

struct ChannelKineticEnergyHistory
{
    std::filesystem::path directory{};
    std::ofstream csv{};

    double runningMean = 0.0;
    double previousEnergy = 0.0;

    std::size_t sampleCount = 0;
    label_t lastStep = 0;
};

// ============================================================
// Initialize output
// ============================================================

inline void initializeChannelKineticEnergyHistory(
    ChannelKineticEnergyHistory &history,
    const std::filesystem::path &directory)
{
    static_assert(sizeof(statistics_t) == sizeof(float),
                  "The existing energy reduction uses FP32");

    history.directory = directory;

    std::filesystem::create_directories(directory);

    history.csv.open(
        directory / "energy_history.csv",
        std::ios::out | std::ios::trunc
    );

    if (!history.csv)
    {
        throw std::runtime_error(
            "Cannot open channel energy_history.csv"
        );
    }

    history.csv
        << "step,t_plus,t_star,energy_sum,"
        << "kinetic_energy_plus,running_mean_plus,"
        << "relative_change\n";

    history.csv << std::setprecision(
        std::numeric_limits<double>::max_digits10
    );

    if (!history.csv)
    {
        throw std::runtime_error(
            "Cannot write channel energy header"
        );
    }
}

// ============================================================
// Record kinetic energy
// ============================================================

inline void recordChannelKineticEnergy(
    ChannelKineticEnergyHistory &history,
    const label_t step,
    const statistics_t energySum)
{
    if (!std::isfinite(energySum) || energySum < 0.0f)
    {
        throw std::runtime_error(
            "Invalid channel kinetic-energy reduction"
        );
    }

    if (history.sampleCount > 0 && step <= history.lastStep)
    {
        throw std::runtime_error(
            "Channel kinetic-energy steps are not increasing"
        );
    }

    // ========================================================
    // Normalization
    //
    // The reduction returns 0.5 * sum(|u|^2).
    //
    // K+ = energySum / (Nphysical * uTau^2)
    // ========================================================

    const double energy = static_cast<double>(energySum);

    const double normalization =
        static_cast<double>(Nphysical) *
        static_cast<double>(channel_uTau) *
        static_cast<double>(channel_uTau);

    if (normalization <= 0.0 ||
        channel_nu <= 0.0f ||
        channel_halfHeight <= 0.0f)
    {
        throw std::runtime_error(
            "Invalid channel kinetic-energy normalization"
        );
    }

    // ========================================================
    // Relative energy change between samples
    // ========================================================

    const double relativeChange =
        history.sampleCount > 0 && history.previousEnergy > 0.0
            ? std::abs(energy - history.previousEnergy) /
                  history.previousEnergy
            : std::numeric_limits<double>::quiet_NaN();

    // ========================================================
    // Running average
    // ========================================================

    ++history.sampleCount;

    history.runningMean +=
        (energy - history.runningMean) /
        static_cast<double>(history.sampleCount);

    history.previousEnergy = energy;
    history.lastStep = step;

    // ========================================================
    // Dimensionless times
    // ========================================================

    const double tPlus =
        static_cast<double>(step) *
        static_cast<double>(channel_uTau) *
        static_cast<double>(channel_uTau) /
        static_cast<double>(channel_nu);

    const double tStar =
        static_cast<double>(step) *
        static_cast<double>(channel_uTau) /
        static_cast<double>(channel_halfHeight);

    // ========================================================
    // CSV output
    // ========================================================

    history.csv
        << step << ','
        << tPlus << ','
        << tStar << ','
        << energy << ','
        << energy / normalization << ','
        << history.runningMean / normalization << ','
        << relativeChange << '\n';

    // Allow plotting during the simulation.
    history.csv.flush();

    if (!history.csv)
    {
        throw std::runtime_error(
            "Failed writing channel kinetic-energy history"
        );
    }
}

// ============================================================
// Close output
// ============================================================

inline void closeChannelKineticEnergyHistory(
    ChannelKineticEnergyHistory &history)
{
    if (history.csv.is_open())
    {
        history.csv.close();

        if (history.csv.fail())
        {
            throw std::runtime_error(
                "Failed closing channel kinetic-energy history"
            );
        }
    }
}

#endif // CHANNEL_ENERGY_STATISTICS_CUH

#ifndef TAYLORGREEN_STATISTICS_CUH
#define TAYLORGREEN_STATISTICS_CUH

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>

#include "../../src/constants.cuh"
#include "../../src/utilities/types.cuh"

// ============================================================
// Taylor-Green time
// ============================================================

[[nodiscard]] inline constexpr statistics_t taylorGreenTimeStar(const label_t step) noexcept
{
    return static_cast<statistics_t>(step) * static_cast<statistics_t>(tgv_U0) * static_cast<statistics_t>(tgv_kx);
}

// ============================================================
// Taylor-Green kinetic-energy normalization
// ============================================================

inline constexpr statistics_t tgvKineticEnergyNormalization = static_cast<statistics_t>(Nphysical) * static_cast<statistics_t>(tgv_U0) * static_cast<statistics_t>(tgv_U0);
inline constexpr statistics_t tgvExpectedKineticEnergyStar = static_cast<statistics_t>(1.0 / 8.0);
inline constexpr statistics_t tgvExpectedKineticEnergySum = tgvKineticEnergyNormalization * tgvExpectedKineticEnergyStar;

static_assert(Nphysical > static_cast<std::size_t>(0));
static_assert(tgv_U0 > static_cast<real_t>(0));

[[nodiscard]] inline constexpr statistics_t taylorGreenKineticEnergyStar(const statistics_t kineticEnergySum) noexcept
{
    return kineticEnergySum / tgvKineticEnergyNormalization;
}

[[nodiscard]] inline constexpr statistics_t taylorGreenInitialKineticEnergyRelativeError(const statistics_t kineticEnergyStar) noexcept
{
    const statistics_t difference = kineticEnergyStar - tgvExpectedKineticEnergyStar;
    return (difference < static_cast<statistics_t>(0) ? -difference : difference) / tgvExpectedKineticEnergyStar;
}

// ============================================================
// Taylor-Green enstrophy normalization
// ============================================================

inline constexpr statistics_t tgvEnstrophyNormalization = static_cast<statistics_t>(Nphysical) * static_cast<statistics_t>(tgv_U0) * static_cast<statistics_t>(tgv_U0) / (static_cast<statistics_t>(tgv_L) * static_cast<statistics_t>(tgv_L));

[[nodiscard]] inline statistics_t taylorGreenEnstrophyStar(const statistics_t enstrophySum) noexcept
{
    return enstrophySum / tgvEnstrophyNormalization;
}

[[nodiscard]] inline statistics_t taylorGreenExpectedInitialEnstrophyStarFD() noexcept
{
    const statistics_t k = static_cast<statistics_t>(tgv_kx);
    const statistics_t modifiedWavenumber = std::sin(k);
    const statistics_t ratio = modifiedWavenumber / k;
    return static_cast<statistics_t>(3.0 / 8.0) * ratio * ratio;
}

[[nodiscard]] inline statistics_t taylorGreenExpectedInitialEnstrophySumFD() noexcept
{
    return tgvEnstrophyNormalization * taylorGreenExpectedInitialEnstrophyStarFD();
}

[[nodiscard]] inline statistics_t taylorGreenInitialEnstrophyRelativeError(const statistics_t enstrophyStar) noexcept
{
    const statistics_t expected = taylorGreenExpectedInitialEnstrophyStarFD();
    const statistics_t difference = enstrophyStar - expected;
    return (difference < static_cast<statistics_t>(0) ? -difference : difference) / expected;
}

// ============================================================
// Binary scalar-statistics format
// ============================================================

inline constexpr std::size_t tgvStatisticsStepOffsetBytes = static_cast<std::size_t>(0);
inline constexpr std::size_t tgvStatisticsValueOffsetBytes = sizeof(label_t);
inline constexpr std::size_t tgvStatisticsRecordBytes = sizeof(label_t) + sizeof(statistics_t);

static_assert(std::is_same_v<label_t, std::uint32_t>);
static_assert(sizeof(label_t) == static_cast<std::size_t>(4));
static_assert(std::is_same_v<statistics_t, float> || std::is_same_v<statistics_t, double>);
static_assert(std::numeric_limits<statistics_t>::is_iec559);
static_assert(std::endian::native == std::endian::little);
static_assert(tgvStatisticsStepOffsetBytes == static_cast<std::size_t>(0));
static_assert(tgvStatisticsValueOffsetBytes == sizeof(label_t));

[[nodiscard]] inline constexpr const char *taylorGreenStatisticsValueDtype() noexcept
{
    if constexpr (std::is_same_v<statistics_t, float>)
    {
        return "float32";
    }
    else
    {
        return "float64";
    }
}

[[nodiscard]] inline constexpr const char *taylorGreenStatisticsValueEncoding() noexcept
{
    if constexpr (std::is_same_v<statistics_t, float>)
    {
        return "ieee754_binary32";
    }
    else
    {
        return "ieee754_binary64";
    }
}

// ============================================================
// Taylor-Green statistics output
// ============================================================

struct TaylorGreenStatisticsOutput
{
    std::ofstream kineticEnergy;
    std::ofstream enstrophy;
};

[[nodiscard]] inline std::filesystem::path taylorGreenStatisticsDirectory()
{
    const std::string reynoldsDirectory = "TG_RE_" + std::to_string(static_cast<label_t>(tgv_Re));
    return std::filesystem::path("post") / "taylorgreen" / "validation" / "data" / reynoldsDirectory;
}

[[nodiscard]] inline std::filesystem::path taylorGreenKineticEnergyStatisticsPath()
{
    return taylorGreenStatisticsDirectory() / "kinetic_energy.bin";
}

[[nodiscard]] inline std::filesystem::path taylorGreenEnstrophyStatisticsPath()
{
    return taylorGreenStatisticsDirectory() / "enstrophy.bin";
}

[[nodiscard]] inline std::filesystem::path taylorGreenStatisticsMetadataPath()
{
    return taylorGreenStatisticsDirectory() / "statistics_metadata.txt";
}

// ============================================================
// Metadata
// ============================================================

inline void writeTaylorGreenStatisticsMetadata()
{
    const std::filesystem::path metadataPath = taylorGreenStatisticsMetadataPath();
    std::ofstream metadata(metadataPath, std::ios::out | std::ios::trunc);

    if (!metadata.is_open())
    {
        throw std::runtime_error("Failed to open Taylor-Green statistics metadata file '" + metadataPath.string() + "'");
    }

    metadata << std::setprecision(std::numeric_limits<statistics_t>::max_digits10)
             << "format_version 2\n"
             << "kinetic_energy_file kinetic_energy.bin\n"
             << "enstrophy_file enstrophy.bin\n"
             << "Re " << tgv_Re << '\n'
             << "NX " << NX << '\n'
             << "NY " << NY << '\n'
             << "NZ " << NZ << '\n'
             << "NXPhysical " << NXPhysical << '\n'
             << "NYPhysical " << NYPhysical << '\n'
             << "NZPhysical " << NZPhysical << '\n'
             << "Nphysical " << Nphysical << '\n'
             << "tgv_U0 " << tgv_U0 << '\n'
             << "tgv_kx " << tgv_kx << '\n'
             << "tgv_ky " << tgv_ky << '\n'
             << "tgv_kz " << tgv_kz << '\n'
             << "tgv_L " << tgv_L << '\n'
             << "Steps " << Steps << '\n'
             << "statisticsStart " << statisticsStart << '\n'
             << "statisticsEvery " << statisticsEvery << '\n'
             << "record_step_dtype uint32\n"
             << "record_value_dtype " << taylorGreenStatisticsValueDtype() << '\n'
             << "record_value_encoding " << taylorGreenStatisticsValueEncoding() << '\n'
             << "record_byte_order little_endian\n"
             << "record_layout step_then_value\n"
             << "record_step_offset_bytes " << tgvStatisticsStepOffsetBytes << '\n'
             << "record_value_offset_bytes " << tgvStatisticsValueOffsetBytes << '\n'
             << "record_step_bytes " << sizeof(label_t) << '\n'
             << "record_value_bytes " << sizeof(statistics_t) << '\n'
             << "record_bytes " << tgvStatisticsRecordBytes << '\n'
             << "kinetic_energy_definition half_velocity_squared_sum\n"
             << "enstrophy_definition half_vorticity_squared_sum\n"
             << "vorticity_derivative central_second_order\n"
             << "grid_spacing 1\n";

    if (!metadata)
    {
        throw std::runtime_error("Failed to write Taylor-Green statistics metadata file '" + metadataPath.string() + "'");
    }

    metadata.close();

    if (metadata.fail())
    {
        throw std::runtime_error("Failed to close Taylor-Green statistics metadata file '" + metadataPath.string() + "'");
    }
}

// ============================================================
// Binary record writer
// ============================================================

inline void writeTaylorGreenScalarRecord(std::ofstream &file, const label_t step, const statistics_t value, const char *quantity)
{
    if (!file.is_open())
    {
        throw std::runtime_error(std::string("Taylor-Green ") + quantity + " statistics file is not open");
    }

    file.write(reinterpret_cast<const char *>(&step), static_cast<std::streamsize>(sizeof(label_t)));

    if (!file)
    {
        throw std::runtime_error(std::string("Failed to write Taylor-Green ") + quantity + " statistics step");
    }

    file.write(reinterpret_cast<const char *>(&value), static_cast<std::streamsize>(sizeof(statistics_t)));

    if (!file)
    {
        throw std::runtime_error(std::string("Failed to write Taylor-Green ") + quantity + " statistics value");
    }
}

// ============================================================
// Initialization
// ============================================================

inline void initializeTaylorGreenStatisticsOutput(TaylorGreenStatisticsOutput &output)
{
    const std::filesystem::path directory = taylorGreenStatisticsDirectory();
    std::error_code error;

    std::filesystem::create_directories(directory, error);

    if (error)
    {
        throw std::runtime_error("Failed to create Taylor-Green statistics directory '" + directory.string() + "': " + error.message());
    }

    writeTaylorGreenStatisticsMetadata();

    output.kineticEnergy.open(taylorGreenKineticEnergyStatisticsPath(), std::ios::out | std::ios::binary | std::ios::trunc);

    if (!output.kineticEnergy.is_open())
    {
        throw std::runtime_error("Failed to open Taylor-Green kinetic-energy statistics file");
    }

    output.enstrophy.open(taylorGreenEnstrophyStatisticsPath(), std::ios::out | std::ios::binary | std::ios::trunc);

    if (!output.enstrophy.is_open())
    {
        output.kineticEnergy.close();
        throw std::runtime_error("Failed to open Taylor-Green enstrophy statistics file");
    }
}

// ============================================================
// Quantity writers
// ============================================================

inline void writeTaylorGreenKineticEnergy(TaylorGreenStatisticsOutput &output, const label_t step, const statistics_t kineticEnergySum)
{
    writeTaylorGreenScalarRecord(output.kineticEnergy, step, kineticEnergySum, "kinetic-energy");
}

inline void writeTaylorGreenEnstrophy(TaylorGreenStatisticsOutput &output, const label_t step, const statistics_t enstrophySum)
{
    writeTaylorGreenScalarRecord(output.enstrophy, step, enstrophySum, "enstrophy");
}

// ============================================================
// Close output
// ============================================================

inline void closeTaylorGreenStatisticsFile(std::ofstream &file, const char *quantity)
{
    if (!file.is_open())
    {
        return;
    }

    file.flush();

    if (!file)
    {
        throw std::runtime_error(std::string("Failed to flush Taylor-Green ") + quantity + " statistics output");
    }

    file.close();

    if (file.fail())
    {
        throw std::runtime_error(std::string("Failed to close Taylor-Green ") + quantity + " statistics output");
    }
}

inline void closeTaylorGreenStatisticsOutput(TaylorGreenStatisticsOutput &output)
{
    closeTaylorGreenStatisticsFile(output.kineticEnergy, "kinetic-energy");
    closeTaylorGreenStatisticsFile(output.enstrophy, "enstrophy");
}

#endif
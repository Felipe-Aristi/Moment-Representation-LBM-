#ifndef TAYLORGREEN_STATISTICS_CUH
#define TAYLORGREEN_STATISTICS_CUH

#include <bit>
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
// Taylor-Green kinetic-energy normalization
// ============================================================

inline constexpr statistics_t tgvKineticEnergyNormalization = static_cast<statistics_t>(Nphysical) * static_cast<statistics_t>(tgv_U0) * static_cast<statistics_t>(tgv_U0);
inline constexpr statistics_t tgvExpectedKineticEnergyStar = static_cast<statistics_t>(1) / static_cast<statistics_t>(8);
inline constexpr statistics_t tgvExpectedKineticEnergySum = tgvKineticEnergyNormalization * tgvExpectedKineticEnergyStar;

static_assert(Nphysical > static_cast<label_t>(0));
static_assert(tgv_U0 > static_cast<real_t>(0));

[[nodiscard]] inline constexpr statistics_t taylorGreenTimeStar(const label_t step) noexcept
{
    return static_cast<statistics_t>(step) * static_cast<statistics_t>(tgv_U0) * static_cast<statistics_t>(tgv_kx);
}

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
// Taylor-Green scalar statistics output
// ============================================================

inline constexpr std::size_t tgvKineticEnergyStepOffsetBytes = static_cast<std::size_t>(0);
inline constexpr std::size_t tgvKineticEnergySumOffsetBytes = sizeof(label_t);
inline constexpr std::size_t tgvKineticEnergyRecordBytes = sizeof(label_t) + sizeof(statistics_t);

static_assert(std::is_same_v<label_t, std::uint32_t>);
static_assert(std::is_same_v<statistics_t, double>);
static_assert(sizeof(label_t) == static_cast<std::size_t>(4));
static_assert(sizeof(statistics_t) == static_cast<std::size_t>(8));
static_assert(std::numeric_limits<statistics_t>::is_iec559);
static_assert(std::numeric_limits<statistics_t>::radix == 2);
static_assert(std::numeric_limits<statistics_t>::digits == 53);
static_assert(std::numeric_limits<statistics_t>::max_exponent == 1024);
static_assert(std::endian::native == std::endian::little);
static_assert(tgvKineticEnergyStepOffsetBytes == static_cast<std::size_t>(0));
static_assert(tgvKineticEnergySumOffsetBytes == static_cast<std::size_t>(4));
static_assert(tgvKineticEnergyRecordBytes == static_cast<std::size_t>(12));

struct TaylorGreenStatisticsOutput
{
    std::ofstream kineticEnergy;
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

[[nodiscard]] inline std::filesystem::path taylorGreenStatisticsMetadataPath()
{
    return taylorGreenStatisticsDirectory() / "statistics_metadata.txt";
}

inline void writeTaylorGreenStatisticsMetadata()
{
    const std::filesystem::path metadataPath = taylorGreenStatisticsMetadataPath();
    std::ofstream metadata(metadataPath, std::ios::out | std::ios::trunc);

    if (!metadata.is_open())
    {
        throw std::runtime_error("Failed to open Taylor-Green statistics metadata file '" + metadataPath.string() + "'");
    }

    metadata << std::setprecision(std::numeric_limits<statistics_t>::max_digits10)
             << "format_version 1\n"
             << "statistics_file kinetic_energy.bin\n"
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
             << "tgv_L " << tgv_L << '\n'
             << "Steps " << Steps << '\n'
             << "statisticsStart " << statisticsStart << '\n'
             << "statisticsEvery " << statisticsEvery << '\n'
             << "record_step_dtype uint32\n"
             << "record_SK_dtype float64\n"
             << "record_SK_encoding ieee754_binary64\n"
             << "record_byte_order little_endian\n"
             << "record_layout step_then_SK\n"
             << "record_step_offset_bytes " << tgvKineticEnergyStepOffsetBytes << '\n'
             << "record_SK_offset_bytes " << tgvKineticEnergySumOffsetBytes << '\n'
             << "record_step_bytes " << sizeof(label_t) << '\n'
             << "record_SK_bytes " << sizeof(statistics_t) << '\n'
             << "record_bytes " << tgvKineticEnergyRecordBytes << '\n';

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

inline void initializeTaylorGreenStatisticsOutput(TaylorGreenStatisticsOutput &output)
{
    const std::filesystem::path filePath = taylorGreenKineticEnergyStatisticsPath();
    std::error_code error;

    std::filesystem::create_directories(filePath.parent_path(), error);

    if (error)
    {
        throw std::runtime_error("Failed to create Taylor-Green statistics directory '" + filePath.parent_path().string() + "': " + error.message());
    }

    writeTaylorGreenStatisticsMetadata();

    output.kineticEnergy.open(filePath, std::ios::out | std::ios::binary | std::ios::trunc);

    if (!output.kineticEnergy.is_open())
    {
        throw std::runtime_error("Failed to open Taylor-Green kinetic-energy statistics file '" + filePath.string() + "'");
    }
}

inline void writeTaylorGreenKineticEnergy(TaylorGreenStatisticsOutput &output, const label_t step, const statistics_t kineticEnergySum)
{
    if (!output.kineticEnergy.is_open())
    {
        throw std::runtime_error("Taylor-Green kinetic-energy statistics file is not open");
    }

    output.kineticEnergy.write(reinterpret_cast<const char *>(&step), static_cast<std::streamsize>(sizeof(label_t)));

    if (!output.kineticEnergy)
    {
        throw std::runtime_error("Failed to write Taylor-Green kinetic-energy statistics step");
    }

    output.kineticEnergy.write(reinterpret_cast<const char *>(&kineticEnergySum), static_cast<std::streamsize>(sizeof(statistics_t)));

    if (!output.kineticEnergy)
    {
        throw std::runtime_error("Failed to write Taylor-Green kinetic-energy statistics sum");
    }
}

inline void closeTaylorGreenStatisticsOutput(TaylorGreenStatisticsOutput &output)
{
    if (!output.kineticEnergy.is_open())
    {
        return;
    }

    output.kineticEnergy.flush();

    if (!output.kineticEnergy)
    {
        throw std::runtime_error("Failed to flush Taylor-Green kinetic-energy statistics output");
    }

    output.kineticEnergy.close();

    if (output.kineticEnergy.fail())
    {
        throw std::runtime_error("Failed to close Taylor-Green kinetic-energy statistics output");
    }
}

#endif

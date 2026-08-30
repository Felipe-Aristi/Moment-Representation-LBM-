#ifndef CHANNELFLOW_VTI_CUH
#define CHANNELFLOW_VTI_CUH

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "../../src/constants.cuh"
#include "../../src/memory.cuh"
#include "../../src/utilities/indexing.cuh"
#include "../../src/utilities/types.cuh"

// ============================================================
// VTK floating-point type
// ============================================================

[[nodiscard]] inline const char *channelVtkRealType() noexcept
{
    if constexpr (sizeof(real_t) == sizeof(float))
    {
        return "Float32";
    }
    else
    {
        return "Float64";
    }
}

// ============================================================
// Channel-flow VTI output
// ============================================================

inline void writeChannelFlowVTI(const HostState &host, const label_t step)
{
    constexpr label_t outNX = NX - static_cast<label_t>(2);
    constexpr label_t outNY = NY;
    constexpr label_t outNZ = NZ - static_cast<label_t>(2);

    constexpr std::size_t outNcells =
        static_cast<std::size_t>(outNX) *
        static_cast<std::size_t>(outNY) *
        static_cast<std::size_t>(outNZ);

    // ========================================================
    // Output directory
    // ========================================================

    std::ostringstream directoryName;
    directoryName << "post/channelflow/CHANNEL_RETAU_"
                  << static_cast<label_t>(channel_ReTau);

    const std::filesystem::path directory(directoryName.str());

    std::filesystem::create_directories(directory);

    // ========================================================
    // Output filename
    // ========================================================

    std::ostringstream fileName;

    fileName << "channel_"
             << std::setw(8)
             << std::setfill('0')
             << step
             << ".vti";

    const std::filesystem::path filePath = directory / fileName.str();

    // ========================================================
    // Temporary contiguous output fields
    // ========================================================

    std::vector<real_t> density(outNcells);
    std::vector<real_t> velocity(static_cast<std::size_t>(3) * outNcells);

    std::size_t outID = static_cast<std::size_t>(0);

    // ========================================================
    // Copy physical channel domain
    //
    // x = 1 ... NX - 2  -> remove periodic ghost planes
    // y = 0 ... NY - 1  -> include both physical walls
    // z = 1 ... NZ - 2  -> remove periodic ghost planes
    // ========================================================

    for (label_t z = static_cast<label_t>(1); z < NZ - static_cast<label_t>(1); ++z)
    {
        for (label_t y = static_cast<label_t>(0); y < NY; ++y)
        {
            for (label_t x = static_cast<label_t>(1); x < NX - static_cast<label_t>(1); ++x)
            {
                const label_t id = idx(x, y, z);

                density[outID] = host.rho[id];

                velocity[static_cast<std::size_t>(3) * outID] = host.ux[id];
                velocity[static_cast<std::size_t>(3) * outID + static_cast<std::size_t>(1)] = host.uy[id];
                velocity[static_cast<std::size_t>(3) * outID + static_cast<std::size_t>(2)] = host.uz[id];

                ++outID;
            }
        }
    }

    // ========================================================
    // Binary block sizes and offsets
    // ========================================================

    const std::uint64_t densityBytes =
        static_cast<std::uint64_t>(density.size() * sizeof(real_t));

    const std::uint64_t velocityBytes =
        static_cast<std::uint64_t>(velocity.size() * sizeof(real_t));

    constexpr std::uint64_t densityOffset = static_cast<std::uint64_t>(0);

    const std::uint64_t velocityOffset =
        static_cast<std::uint64_t>(sizeof(std::uint64_t)) +
        densityBytes;

    // ========================================================
    // Write VTI
    // ========================================================

    std::ofstream file(filePath, std::ios::binary);

    file << "<?xml version=\"1.0\"?>\n";

    file << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";

    file << "  <ImageData WholeExtent=\"0 "
         << outNX - static_cast<label_t>(1)
         << " 0 "
         << outNY - static_cast<label_t>(1)
         << " 0 "
         << outNZ - static_cast<label_t>(1)
         << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";

    file << "    <Piece Extent=\"0 "
         << outNX - static_cast<label_t>(1)
         << " 0 "
         << outNY - static_cast<label_t>(1)
         << " 0 "
         << outNZ - static_cast<label_t>(1)
         << "\">\n";

    file << "      <PointData Scalars=\"Density\" Vectors=\"Velocity\">\n";

    file << "        <DataArray type=\""
         << channelVtkRealType()
         << "\" Name=\"Density\" NumberOfComponents=\"1\" format=\"appended\" offset=\""
         << densityOffset
         << "\"/>\n";

    file << "        <DataArray type=\""
         << channelVtkRealType()
         << "\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"appended\" offset=\""
         << velocityOffset
         << "\"/>\n";

    file << "      </PointData>\n";

    file << "      <CellData/>\n";

    file << "    </Piece>\n";

    file << "  </ImageData>\n";

    file << "  <AppendedData encoding=\"raw\">\n_";

    file.write(reinterpret_cast<const char *>(&densityBytes), sizeof(std::uint64_t));

    file.write(reinterpret_cast<const char *>(density.data()),
               static_cast<std::streamsize>(densityBytes));

    file.write(reinterpret_cast<const char *>(&velocityBytes), sizeof(std::uint64_t));

    file.write(reinterpret_cast<const char *>(velocity.data()),
               static_cast<std::streamsize>(velocityBytes));

    file << "\n  </AppendedData>\n";

    file << "</VTKFile>\n";
}

#endif
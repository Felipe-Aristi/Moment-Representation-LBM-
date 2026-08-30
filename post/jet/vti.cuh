#ifndef JET_VTI_CUH
#define JET_VTI_CUH

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
// Jet VTI constants
// ============================================================

inline constexpr label_t jetVtiNX = NX - static_cast<label_t>(2);
inline constexpr label_t jetVtiNY = NY - static_cast<label_t>(2);
inline constexpr label_t jetVtiNZ = NZ - static_cast<label_t>(2);

inline constexpr std::size_t jetVtiNcells = static_cast<std::size_t>(jetVtiNX) * static_cast<std::size_t>(jetVtiNY) * static_cast<std::size_t>(jetVtiNZ);

inline constexpr std::size_t jetVtiRhoValues = jetVtiNcells;
inline constexpr std::size_t jetVtiVelocityValues = static_cast<std::size_t>(3) * jetVtiNcells;

inline constexpr std::uint64_t jetVtiRhoBytes = static_cast<std::uint64_t>(jetVtiRhoValues * sizeof(real_t));
inline constexpr std::uint64_t jetVtiVelocityBytes = static_cast<std::uint64_t>(jetVtiVelocityValues * sizeof(real_t));

inline constexpr std::uint64_t jetVtiRhoOffset = static_cast<std::uint64_t>(0);
inline constexpr std::uint64_t jetVtiVelocityOffset = static_cast<std::uint64_t>(sizeof(std::uint64_t)) + jetVtiRhoBytes;

inline constexpr const char *jetVtkRealType = sizeof(real_t) == sizeof(float) ? "Float32" : "Float64";

// ============================================================
// Jet VTI output
// ============================================================

inline void writeJetVTI(const HostState &state, const label_t step)
{
    std::vector<real_t> rhoData(jetVtiRhoValues);
    std::vector<real_t> velocityData(jetVtiVelocityValues);

    std::size_t outputID = static_cast<std::size_t>(0);

    // ========================================================
    // Physical domain
    // Ghost layers are excluded
    // ========================================================

    for (label_t z = static_cast<label_t>(1); z < NZ - static_cast<label_t>(1); ++z)
    {
        for (label_t y = static_cast<label_t>(1); y < NY - static_cast<label_t>(1); ++y)
        {
            for (label_t x = static_cast<label_t>(1); x < NX - static_cast<label_t>(1); ++x)
            {
                const label_t id = idx(x, y, z);

                rhoData[outputID] = state.rho[id];

                velocityData[static_cast<std::size_t>(3) * outputID + static_cast<std::size_t>(0)] = state.ux[id];
                velocityData[static_cast<std::size_t>(3) * outputID + static_cast<std::size_t>(1)] = state.uy[id];
                velocityData[static_cast<std::size_t>(3) * outputID + static_cast<std::size_t>(2)] = state.uz[id];

                ++outputID;
            }
        }
    }

    // ========================================================
    // Output folder
    // ========================================================

    std::ostringstream folder;
    folder << "post/jet/JET_RE_" << static_cast<label_t>(jet_Re);

    std::filesystem::create_directories(folder.str());

    // ========================================================
    // Output filename
    // ========================================================

    std::ostringstream filename;
    filename << folder.str() << "/jet_" << std::setw(8) << std::setfill('0') << step << ".vti";

    // ========================================================
    // Open file
    // ========================================================

    std::ofstream file(filename.str(), std::ios::binary);

    if (!file)
    {
        return;
    }

    // ========================================================
    // VTI header
    // ========================================================

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";

    file << "  <ImageData WholeExtent=\"0 " << jetVtiNX - static_cast<label_t>(1)
         << " 0 " << jetVtiNY - static_cast<label_t>(1)
         << " 0 " << jetVtiNZ - static_cast<label_t>(1)
         << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";

    file << "    <Piece Extent=\"0 " << jetVtiNX - static_cast<label_t>(1)
         << " 0 " << jetVtiNY - static_cast<label_t>(1)
         << " 0 " << jetVtiNZ - static_cast<label_t>(1)
         << "\">\n";

    // ========================================================
    // Point data
    // ========================================================

    file << "      <PointData Scalars=\"rho\" Vectors=\"velocity\">\n";

    file << "        <DataArray type=\"" << jetVtkRealType
         << "\" Name=\"rho\" NumberOfComponents=\"1\" format=\"appended\" offset=\""
         << jetVtiRhoOffset << "\"/>\n";

    file << "        <DataArray type=\"" << jetVtkRealType
         << "\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"appended\" offset=\""
         << jetVtiVelocityOffset << "\"/>\n";

    file << "      </PointData>\n";

    file << "      <CellData/>\n";
    file << "    </Piece>\n";
    file << "  </ImageData>\n";

    // ========================================================
    // Binary appended data
    // ========================================================

    file << "  <AppendedData encoding=\"raw\">\n_";

    file.write(reinterpret_cast<const char *>(&jetVtiRhoBytes), sizeof(std::uint64_t));
    file.write(reinterpret_cast<const char *>(rhoData.data()), static_cast<std::streamsize>(jetVtiRhoBytes));

    file.write(reinterpret_cast<const char *>(&jetVtiVelocityBytes), sizeof(std::uint64_t));
    file.write(reinterpret_cast<const char *>(velocityData.data()), static_cast<std::streamsize>(jetVtiVelocityBytes));

    // ========================================================
    // Close file
    // ========================================================

    file << "\n  </AppendedData>\n";
    file << "</VTKFile>\n";
}

#endif
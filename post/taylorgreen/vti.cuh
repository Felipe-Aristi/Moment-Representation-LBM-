#ifndef TAYLORGREEN_VTI_CUH
#define TAYLORGREEN_VTI_CUH

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "../../src/constants.cuh"
#include "../../cases/taylorgreen/case_name.cuh"
#include "../../src/memory.cuh"
#include "../../src/utilities/indexing.cuh"
#include "../../src/utilities/types.cuh"

// ============================================================
// Taylor-Green VTI constants
// ============================================================

inline constexpr label_t vtiNX = NX - static_cast<label_t>(2);
inline constexpr label_t vtiNY = NY - static_cast<label_t>(2);
inline constexpr label_t vtiNZ = NZ - static_cast<label_t>(2);

inline constexpr std::size_t vtiNcells = static_cast<std::size_t>(vtiNX) * static_cast<std::size_t>(vtiNY) * static_cast<std::size_t>(vtiNZ);

inline constexpr std::size_t vtiRhoValues = vtiNcells;
inline constexpr std::size_t vtiVelocityValues = static_cast<std::size_t>(3) * vtiNcells;

inline constexpr std::uint64_t vtiRhoBytes = static_cast<std::uint64_t>(vtiRhoValues * sizeof(real_t));
inline constexpr std::uint64_t vtiVelocityBytes = static_cast<std::uint64_t>(vtiVelocityValues * sizeof(real_t));

inline constexpr std::uint64_t vtiRhoOffset = static_cast<std::uint64_t>(0);
inline constexpr std::uint64_t vtiVelocityOffset = static_cast<std::uint64_t>(sizeof(std::uint64_t)) + vtiRhoBytes;

inline constexpr const char *vtkRealType = sizeof(real_t) == sizeof(float) ? "Float32" : "Float64";

// ============================================================
// Taylor-Green VTI output
// ============================================================

inline void writeTaylorGreenVTI(const HostState &state, const label_t step)
{
    std::vector<real_t> rhoData(vtiRhoValues);
    std::vector<real_t> velocityData(vtiVelocityValues);

    std::size_t outputID = static_cast<std::size_t>(0);

    // ========================================================
    // Copy physical domain
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
    folder << "post/taylorgreen/" << taylorGreenCaseName();

    std::filesystem::create_directories(folder.str());

    // ========================================================
    // Output filename
    // ========================================================

    std::ostringstream filename;
    filename << folder.str() << "/tgv_" << std::setw(8) << std::setfill('0') << step << ".vti";

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

    file << "  <ImageData WholeExtent=\"0 " << vtiNX - static_cast<label_t>(1)
         << " 0 " << vtiNY - static_cast<label_t>(1)
         << " 0 " << vtiNZ - static_cast<label_t>(1)
         << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";

    file << "    <Piece Extent=\"0 " << vtiNX - static_cast<label_t>(1)
         << " 0 " << vtiNY - static_cast<label_t>(1)
         << " 0 " << vtiNZ - static_cast<label_t>(1)
         << "\">\n";

    // ========================================================
    // Point data
    // ========================================================

    file << "      <PointData Scalars=\"rho\" Vectors=\"velocity\">\n";

    file << "        <DataArray type=\"" << vtkRealType
         << "\" Name=\"rho\" NumberOfComponents=\"1\" format=\"appended\" offset=\""
         << vtiRhoOffset << "\"/>\n";

    file << "        <DataArray type=\"" << vtkRealType
         << "\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"appended\" offset=\""
         << vtiVelocityOffset << "\"/>\n";

    file << "      </PointData>\n";

    file << "      <CellData/>\n";
    file << "    </Piece>\n";
    file << "  </ImageData>\n";

    // ========================================================
    // Binary appended data
    // ========================================================

    file << "  <AppendedData encoding=\"raw\">\n_";

    file.write(reinterpret_cast<const char *>(&vtiRhoBytes), sizeof(std::uint64_t));
    file.write(reinterpret_cast<const char *>(rhoData.data()), static_cast<std::streamsize>(vtiRhoBytes));

    file.write(reinterpret_cast<const char *>(&vtiVelocityBytes), sizeof(std::uint64_t));
    file.write(reinterpret_cast<const char *>(velocityData.data()), static_cast<std::streamsize>(vtiVelocityBytes));

    // ========================================================
    // Close VTI
    // ========================================================

    file << "\n  </AppendedData>\n";
    file << "</VTKFile>\n";
}

#endif
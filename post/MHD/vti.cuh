#ifndef MHD_VTI_CUH
#define MHD_VTI_CUH

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "../../cases/MHD/constantsmhd.cuh"
#include "../../cases/MHD/magneticMemory.cuh"

#include "../../src/constants.cuh"
#include "../../src/memory.cuh"
#include "../../src/utilities/indexing.cuh"
#include "../../src/utilities/types.cuh"

// ============================================================
// MHD VTI constants
// ============================================================

namespace mhdvti
{
    inline constexpr label_t nx = NX - static_cast<label_t>(2);
    inline constexpr label_t ny = NY - static_cast<label_t>(2);
    inline constexpr label_t nz = NZ - static_cast<label_t>(2);

    inline constexpr std::size_t ncells = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);

    inline constexpr std::size_t rhoValues = ncells;
    inline constexpr std::size_t velocityValues = static_cast<std::size_t>(3) * ncells;
    inline constexpr std::size_t magneticValues = static_cast<std::size_t>(3) * ncells;

    inline constexpr std::uint64_t rhoBytes = static_cast<std::uint64_t>(rhoValues * sizeof(real_t));
    inline constexpr std::uint64_t velocityBytes = static_cast<std::uint64_t>(velocityValues * sizeof(real_t));
    inline constexpr std::uint64_t magneticBytes = static_cast<std::uint64_t>(magneticValues * sizeof(real_t));

    inline constexpr std::uint64_t rhoOffset = static_cast<std::uint64_t>(0);
    inline constexpr std::uint64_t velocityOffset = rhoOffset + static_cast<std::uint64_t>(sizeof(std::uint64_t)) + rhoBytes;
    inline constexpr std::uint64_t magneticOffset = velocityOffset + static_cast<std::uint64_t>(sizeof(std::uint64_t)) + velocityBytes;

    inline constexpr const char *vtkRealType = sizeof(real_t) == sizeof(float) ? "Float32" : "Float64";
}

// ============================================================
// Orszag-Tang VTI output
// ============================================================

inline void writeOrszagTangVTI(const HostState &fluid, const HostMagneticState &magnetic, const label_t step)
{
    std::vector<real_t> rhoData(mhdvti::rhoValues);
    std::vector<real_t> velocityData(mhdvti::velocityValues);
    std::vector<real_t> magneticData(mhdvti::magneticValues);

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

                rhoData[outputID] = fluid.rho[id];

                velocityData[static_cast<std::size_t>(3) * outputID + static_cast<std::size_t>(0)] = fluid.ux[id];
                velocityData[static_cast<std::size_t>(3) * outputID + static_cast<std::size_t>(1)] = fluid.uy[id];
                velocityData[static_cast<std::size_t>(3) * outputID + static_cast<std::size_t>(2)] = fluid.uz[id];

                magneticData[static_cast<std::size_t>(3) * outputID + static_cast<std::size_t>(0)] = magnetic.bx[id];
                magneticData[static_cast<std::size_t>(3) * outputID + static_cast<std::size_t>(1)] = magnetic.by[id];
                magneticData[static_cast<std::size_t>(3) * outputID + static_cast<std::size_t>(2)] = magnetic.bz[id];

                ++outputID;
            }
        }
    }

    // ========================================================
    // Output folder
    // ========================================================

    std::ostringstream folder;
    folder << "post/MHD/OT_RE_" << static_cast<label_t>(ot_Re) << "_RM_" << static_cast<label_t>(ot_Rm);

    std::filesystem::create_directories(folder.str());

    // ========================================================
    // Output filename
    // ========================================================

    std::ostringstream filename;
    filename << folder.str() << "/ot_" << std::setw(8) << std::setfill('0') << step << ".vti";

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

    file << "  <ImageData WholeExtent=\"0 " << mhdvti::nx - static_cast<label_t>(1)
         << " 0 " << mhdvti::ny - static_cast<label_t>(1)
         << " 0 " << mhdvti::nz - static_cast<label_t>(1)
         << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";

    file << "    <Piece Extent=\"0 " << mhdvti::nx - static_cast<label_t>(1)
         << " 0 " << mhdvti::ny - static_cast<label_t>(1)
         << " 0 " << mhdvti::nz - static_cast<label_t>(1)
         << "\">\n";

    // ========================================================
    // Point data
    // ========================================================

    file << "      <PointData Scalars=\"rho\" Vectors=\"velocity\">\n";

    file << "        <DataArray type=\"" << mhdvti::vtkRealType
         << "\" Name=\"rho\" NumberOfComponents=\"1\" format=\"appended\" offset=\""
         << mhdvti::rhoOffset << "\"/>\n";

    file << "        <DataArray type=\"" << mhdvti::vtkRealType
         << "\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"appended\" offset=\""
         << mhdvti::velocityOffset << "\"/>\n";

    file << "        <DataArray type=\"" << mhdvti::vtkRealType
         << "\" Name=\"B\" NumberOfComponents=\"3\" format=\"appended\" offset=\""
         << mhdvti::magneticOffset << "\"/>\n";

    file << "      </PointData>\n";

    file << "      <CellData/>\n";
    file << "    </Piece>\n";
    file << "  </ImageData>\n";

    // ========================================================
    // Binary appended data
    // ========================================================

    file << "  <AppendedData encoding=\"raw\">\n_";

    file.write(reinterpret_cast<const char *>(&mhdvti::rhoBytes), sizeof(std::uint64_t));
    file.write(reinterpret_cast<const char *>(rhoData.data()), static_cast<std::streamsize>(mhdvti::rhoBytes));

    file.write(reinterpret_cast<const char *>(&mhdvti::velocityBytes), sizeof(std::uint64_t));
    file.write(reinterpret_cast<const char *>(velocityData.data()), static_cast<std::streamsize>(mhdvti::velocityBytes));

    file.write(reinterpret_cast<const char *>(&mhdvti::magneticBytes), sizeof(std::uint64_t));
    file.write(reinterpret_cast<const char *>(magneticData.data()), static_cast<std::streamsize>(mhdvti::magneticBytes));

    // ========================================================
    // Close VTI
    // ========================================================

    file << "\n  </AppendedData>\n";
    file << "</VTKFile>\n";
}

#endif
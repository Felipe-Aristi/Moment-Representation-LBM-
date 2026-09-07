#ifndef MHD_OUTPUT_CUH
#define MHD_OUTPUT_CUH

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "memory.cuh"
#include "../../src/utilities/bounds.cuh"

namespace mhd
{
struct Snapshot
{
    std::vector<real_t> values;
    Snapshot() : values(7 * fieldStride) {}
    const real_t *field(const std::size_t a) const { return values.data() + a * fieldStride; }
    real_t *field(const std::size_t a) { return values.data() + a * fieldStride; }
};

inline void copySnapshot(Snapshot &h, const State &s)
{
    CUDA_CHECK(cudaMemcpy(h.values.data(), s.fluid.data, 4 * bytesField, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.field(4), s.data, 3 * bytesField, cudaMemcpyDeviceToHost));
}

struct Derivatives
{
    std::array<double, 3> current;
    std::array<double, 3> vorticity;
    double divergence;
};

inline Derivatives derivatives(const Snapshot &h, const label_t x, const label_t y, const label_t z)
{
    const label_t plus[3] = {idx(wrapx(x + 1), y, z), idx(x, wrapy(y + 1), z), idx(x, y, wrapz(z + 1))};
    const label_t minus[3] = {idx(wrapx(x - 1), y, z), idx(x, wrapy(y - 1), z), idx(x, y, wrapz(z - 1))};
    double du[3][3], db[3][3];
    for (int a = 0; a < 3; ++a)
        for (int j = 0; j < 3; ++j)
        {
            du[a][j] = 0.5 * (static_cast<double>(h.field(1 + a)[plus[j]]) - h.field(1 + a)[minus[j]]);
            db[a][j] = 0.5 * (static_cast<double>(h.field(4 + a)[plus[j]]) - h.field(4 + a)[minus[j]]);
        }
    return {{db[2][1] - db[1][2], db[0][2] - db[2][0], db[1][0] - db[0][1]},
            {du[2][1] - du[1][2], du[0][2] - du[2][0], du[1][0] - du[0][1]},
            db[0][0] + db[1][1] + db[2][2]};
}

inline double norm2(const std::array<double, 3> &v)
{
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

struct Statistics
{
    double mass = 0, kinetic = 0, magnetic = 0, enstrophy = 0, current2 = 0;
    double divRms = 0, divMax = 0, maxMach = 0;
    double rhoMin = std::numeric_limits<double>::infinity(), rhoMax = 0;
    std::array<double, 3> momentum{}, meanB{};
};

// Double-precision accumulation over physical nodes, excluding all ghost planes.
// Host diagnostics are a correctness-first baseline; they are outside the kernel.
inline Statistics statistics(const Snapshot &h)
{
    Statistics s;
    for (label_t z = 1; z < NZ - 1; ++z)
        for (label_t y = 1; y < NY - 1; ++y)
            for (label_t x = 1; x < NX - 1; ++x)
            {
                const label_t id = idx(x, y, z);
                for (int a = 0; a < 7; ++a)
                    if (!std::isfinite(h.field(a)[id])) throw std::runtime_error("Non-finite MHD field detected.");
                const double rho = h.field(0)[id];
                if (rho <= 0) throw std::runtime_error("Non-positive MHD density detected.");
                std::array<double, 3> u{}, b{};
                for (int a = 0; a < 3; ++a)
                {
                    u[a] = h.field(1 + a)[id]; b[a] = h.field(4 + a)[id];
                    s.momentum[a] += rho * u[a]; s.meanB[a] += b[a];
                }
                const auto d = derivatives(h, x, y, z);
                s.mass += rho;
                s.kinetic += 0.5 * rho * norm2(u);
                s.magnetic += 0.5 * norm2(b);
                s.enstrophy += 0.5 * norm2(d.vorticity);
                s.current2 += norm2(d.current);
                s.divRms += d.divergence * d.divergence;
                s.divMax = std::max(s.divMax, std::abs(d.divergence));
                s.maxMach = std::max(s.maxMach, std::sqrt(norm2(u) / cs2));
                s.rhoMin = std::min(s.rhoMin, rho); s.rhoMax = std::max(s.rhoMax, rho);
            }
    const double count = static_cast<double>(Nphysical);
    s.kinetic /= count; s.magnetic /= count; s.enstrophy /= count; s.current2 /= count;
    s.divRms = std::sqrt(s.divRms / count);
    for (int a = 0; a < 3; ++a) { s.momentum[a] /= count; s.meanB[a] /= count; }
    return s;
}

inline double timeStar(const label_t step)
{
    return static_cast<double>(step) * velocity / length;
}

inline void writeStatisticsHeader(std::ostream &out)
{
    out << "step,tStar,mass,K,M,E,enstrophy,meanJ2,divB_rms,divB_max,divB_relative,maxMach,rhoMin,rhoMax,meanRhoUx,meanRhoUy,meanRhoUz,meanBx,meanBy,meanBz\n";
}

inline void writeStatistics(std::ostream &out, const label_t step, const Statistics &s)
{
    out << std::setprecision(17) << step << ',' << timeStar(step) << ',' << s.mass << ',' << s.kinetic << ',' << s.magnetic << ',' << s.kinetic + s.magnetic
        << ',' << s.enstrophy << ',' << s.current2 << ',' << s.divRms << ',' << s.divMax
        << ',' << length * s.divRms / std::max(std::sqrt(2 * s.magnetic), 1e-30) << ',' << s.maxMach << ',' << s.rhoMin << ',' << s.rhoMax;
    for (double p : s.momentum) out << ',' << p;
    for (double b : s.meanB) out << ',' << b;
    out << '\n';
    out.flush();
    if (!out) throw std::runtime_error("Failed to write MHD statistics.");
}

inline std::string snapshotName(const label_t step)
{
    return "fields_" + std::to_string(step) + ".vti";
}

inline void writeVTI(const std::filesystem::path &directory, const Snapshot &h, const label_t step)
{
    std::ofstream out(directory / snapshotName(step));
    if (!out) throw std::runtime_error("Cannot open MHD VTI output.");
    out << std::setprecision(std::numeric_limits<real_t>::max_digits10);
    out << "<?xml version=\"1.0\"?>\n<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "<ImageData WholeExtent=\"0 " << nx - 1 << " 0 " << ny - 1 << " 0 " << nz - 1 << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";
    out << "<Piece Extent=\"0 " << nx - 1 << " 0 " << ny - 1 << " 0 " << nz - 1 << "\"><PointData Scalars=\"rho\" Vectors=\"velocity\">\n";
    auto array = [&](const char *name, const int components, auto value)
    {
        out << "<DataArray type=\"Float32\" Name=\"" << name << "\" NumberOfComponents=\"" << components << "\" format=\"ascii\">\n";
        for (label_t z = 1; z < NZ - 1; ++z)
            for (label_t y = 1; y < NY - 1; ++y)
                for (label_t x = 1; x < NX - 1; ++x) value(x, y, z);
        out << "\n</DataArray>\n";
    };
    array("rho", 1, [&](label_t x, label_t y, label_t z) { out << h.field(0)[idx(x, y, z)] << ' '; });
    array("velocity", 3, [&](label_t x, label_t y, label_t z)
    {
        for (int a = 1; a < 4; ++a) out << h.field(a)[idx(x, y, z)] << ' ';
    });
    array("B", 3, [&](label_t x, label_t y, label_t z)
    {
        for (int a = 4; a < 7; ++a) out << h.field(a)[idx(x, y, z)] << ' ';
    });
    array("current", 3, [&](label_t x, label_t y, label_t z)
    {
        for (double j : derivatives(h, x, y, z).current) out << j << ' ';
    });
    array("divB", 1, [&](label_t x, label_t y, label_t z) { out << derivatives(h, x, y, z).divergence << ' '; });
    array("kinetic_energy", 1, [&](label_t x, label_t y, label_t z)
    {
        const auto id = idx(x, y, z);
        const std::array<double, 3> u = {h.field(1)[id], h.field(2)[id], h.field(3)[id]};
        out << 0.5 * h.field(0)[id] * norm2(u) << ' ';
    });
    array("magnetic_energy", 1, [&](label_t x, label_t y, label_t z)
    {
        const auto id = idx(x, y, z);
        const std::array<double, 3> b = {h.field(4)[id], h.field(5)[id], h.field(6)[id]};
        out << 0.5 * norm2(b) << ' ';
    });
    out << "</PointData><CellData/></Piece></ImageData></VTKFile>\n";
    out.close();
    if (!out) throw std::runtime_error("Failed to write MHD VTI output.");
}

inline void writePVD(const std::filesystem::path &directory, const std::vector<label_t> &stepsWritten)
{
    std::ofstream out(directory / "fields.pvd");
    out << std::setprecision(17) << "<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\"><Collection>\n";
    for (const label_t step : stepsWritten)
        out << "<DataSet timestep=\"" << timeStar(step) << "\" group=\"\" part=\"0\" file=\"" << snapshotName(step) << "\"/>\n";
    out << "</Collection></VTKFile>\n";
    out.close();
    if (!out) throw std::runtime_error("Failed to write MHD PVD output.");
}

inline void writeMetadata(const std::filesystem::path &directory)
{
    std::ofstream out(directory / "metadata.json");
    out << std::setprecision(17)
        << "{\n  \"case\": \"planar low-Mach Orszag-Tang\",\n  \"fluid\": \"D3Q27 MR-R3\",\n  \"magnetic\": \"D3Q7 vector BGK\",\n"
        << "  \"nx\": " << nx << ", \"ny\": " << ny << ", \"nz\": " << nz << ",\n"
        << "  \"steps\": " << steps << ", \"U0\": " << velocity << ", \"B0\": " << magneticAmplitude << ", \"rho0\": " << rho0 << ", \"mu0\": 1,\n"
        << "  \"L\": " << length << ", \"Re\": " << reynolds << ", \"Rm\": " << magneticReynolds << ",\n"
        << "  \"nu\": " << viscosity << ", \"eta\": " << diffusivity << ", \"tau\": " << tau << ", \"tauB\": " << magneticTau << ",\n"
        << "  \"gpu_state_bytes\": " << 2 * (bytesMoments + magneticBytes) << ",\n"
        << "  \"K_definition\": \"mean(rho * |u|^2 / 2)\", \"M_definition\": \"mean(|B|^2 / 2)\",\n"
        << "  \"divB_definition\": \"centered periodic finite difference, lattice units\",\n"
        << "  \"initialization\": \"equilibrium populations with leading incompressible pressure balance\"\n}\n";
    out.close();
    if (!out) throw std::runtime_error("Failed to write MHD metadata.");
}
}
#endif

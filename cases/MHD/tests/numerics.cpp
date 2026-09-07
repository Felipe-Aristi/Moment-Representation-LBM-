#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include "../initialization.cuh"
#include "../output.cuh"

namespace
{
void require(const bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

void near(const double got, const double expected, const double tolerance, const char *message)
{
    if (!std::isfinite(got) || std::abs(got - expected) > tolerance)
    {
        std::cerr << message << ": got " << got << ", expected " << expected << '\n';
        throw std::runtime_error(message);
    }
}

template <typename Function>
void nodes(Function function)
{
    for (label_t z = 1; z < NZ - 1; ++z)
        for (label_t y = 1; y < NY - 1; ++y)
            for (label_t x = 1; x < NX - 1; ++x) function(x, y, z);
}

struct Buffers
{
    mhd::State a{}, b{};
    Buffers()
    {
        mhd::allocate(a); mhd::allocate(b);
        for (auto *s : {&a, &b})
        {
            // NaN ghosts/padding catch accidental reads from nonphysical nodes.
            std::fill_n(s->data, mhd::magneticFields * fieldStride, std::numeric_limits<real_t>::quiet_NaN());
            std::fill_n(s->fluid.data, Nmoments * fieldStride, std::numeric_limits<real_t>::quiet_NaN());
        }
    }
    ~Buffers() { mhd::release(a); mhd::release(b); }
    void advance()
    {
        nodes([&](auto x, auto y, auto z) { mhd::advanceNode(x, y, z, a, b); });
        std::swap(a, b);
    }
};

void setEquilibrium(const mhd::State &s, const label_t id, const real_t rho, const mhd::Vector u, const mhd::Vector b)
{
    LocalState f{};
    f.rho = rho; f.ux = u.x; f.uy = u.y; f.uz = u.z;
    mhd::storeFluid(f, s.fluid, id);
    s.bx[id] = b.x; s.by[id] = b.y; s.bz[id] = b.z;
    constexpr_for<0, 7>([&](auto I)
    {
        constexpr label_t i = decltype(I)::value;
        constexpr_for<0, 3>([&](auto A)
        {
            constexpr label_t alpha = decltype(A)::value;
            s.g[mhd::populationIndex<i, alpha>(id)] = mhd::geq<i, alpha>(u, b);
        });
    });
}

void checkMoments()
{
    std::mt19937 random(8147);
    std::uniform_real_distribution<float> dist(-0.12f, 0.12f);
    for (int sample = 0; sample < 16; ++sample)
    {
        const mhd::Vector u{dist(random), dist(random), dist(random)};
        const mhd::Vector b{dist(random), dist(random), dist(random)};
        const real_t rho = 1 + dist(random);
        const double bv[] = {b.x, b.y, b.z}, uv[] = {u.x, u.y, u.z};
        double delta0 = 0, delta1[3]{}, delta2[3][3]{};
        double eq0 = 0, eq1[3]{}, eq2[3][3]{};
        constexpr_for<0, Q>([&](auto I)
        {
            constexpr label_t i = decltype(I)::value;
            const int c[] = {D3Q27::cx<i>(), D3Q27::cy<i>(), D3Q27::cz<i>()};
            const double delta = mhd::magneticEquilibrium<i>(b);
            const double eq = mhd::feqMHD<i>(rho, u, b);
            delta0 += delta; eq0 += eq;
            for (int a = 0; a < 3; ++a)
            {
                delta1[a] += c[a] * delta; eq1[a] += c[a] * eq;
                for (int j = 0; j < 3; ++j) { delta2[a][j] += c[a] * c[j] * delta; eq2[a][j] += c[a] * c[j] * eq; }
            }
        });
        const double pressure = 0.5 * (b.x * b.x + b.y * b.y + b.z * b.z);
        near(delta0, 0, 2e-8, "magnetic equilibrium adds mass");
        near(eq0, rho, 5e-7, "MHD equilibrium mass");
        for (int a = 0; a < 3; ++a)
        {
            near(delta1[a], 0, 2e-8, "magnetic equilibrium adds momentum");
            near(eq1[a], rho * uv[a], 5e-8, "MHD equilibrium momentum");
            for (int j = 0; j < 3; ++j)
            {
                const double m = (a == j ? pressure : 0) - bv[a] * bv[j];
                near(delta2[a][j], m, 2e-8, "Maxwell stress moment");
                near(eq2[a][j], rho * uv[a] * uv[j] + (a == j ? rho * cs2 : 0) + m, 1e-7, "MHD equilibrium stress");
            }
        }
        constexpr_for<0, 3>([&](auto A)
        {
            constexpr label_t alpha = decltype(A)::value;
            double sum = 0, first[3]{}, second[3][3]{};
            constexpr_for<0, 7>([&](auto I)
            {
                constexpr label_t i = decltype(I)::value;
                const int c[] = {mhd::D3Q7::cx<i>(), mhd::D3Q7::cy<i>(), mhd::D3Q7::cz<i>()};
                const double g = mhd::geq<i, alpha>(u, b);
                sum += g;
                for (int j = 0; j < 3; ++j)
                {
                    first[j] += c[j] * g;
                    for (int l = 0; l < 3; ++l) second[j][l] += c[j] * c[l] * g;
                }
            });
            near(sum, bv[alpha], 2e-8, "magnetic zeroth moment");
            for (int j = 0; j < 3; ++j)
            {
                near(first[j], uv[j] * bv[alpha] - bv[j] * uv[alpha], 1e-8, "induction flux orientation");
                for (int l = 0; l < 3; ++l)
                    near(second[j][l], j == l ? mhd::magneticCs2 * bv[alpha] : 0, 1e-8, "magnetic second moment");
            }
        });
    }
    std::cout << "PASS equilibrium moments (16 arbitrary 3D states)\n";
}

void checkUniform()
{
    Buffers s;
    const mhd::Vector u{0.013f, -0.009f, 0.006f}, b{0.035f, 0.027f, -0.016f};
    nodes([&](auto x, auto y, auto z) { setEquilibrium(s.a, idx(x, y, z), 1.03f, u, b); });
    for (int step = 0; step < 20; ++step) s.advance();
    nodes([&](auto x, auto y, auto z)
    {
        const auto id = idx(x, y, z);
        near(s.a.fluid.rho[id], 1.03, 5e-6, "uniform density");
        near(s.a.fluid.ux[id], u.x, 1e-6, "uniform ux");
        near(s.a.fluid.uy[id], u.y, 1e-6, "uniform uy");
        near(s.a.fluid.uz[id], u.z, 1e-6, "uniform uz");
        near(s.a.bx[id], b.x, 2e-7, "uniform Bx");
        near(s.a.by[id], b.y, 2e-7, "uniform By");
        near(s.a.bz[id], b.z, 2e-7, "uniform Bz");
        for (label_t field = 4; field < 10; ++field)
            near(s.a.fluid.data[field * fieldStride + id], 0, 1e-6, "uniform nonequilibrium stress");
    });
    std::cout << "PASS uniform coupled state, periodic streaming, poisoned ghosts\n";
}

void checkMagneticStreaming()
{
    Buffers s;
    // B has zero sum at each node, but each g_i has an independent kinetic mode.
    // This checks all three component addresses and every axial pull direction.
    nodes([&](auto x, auto y, auto z)
    {
        const auto id = idx(x, y, z);
        setEquilibrium(s.a, id, 1, {0, 0, 0}, {0, 0, 0});
        constexpr_for<0, 7>([&](auto I)
        {
            constexpr label_t i = decltype(I)::value;
            constexpr_for<0, 3>([&](auto A)
            {
                constexpr label_t alpha = decltype(A)::value;
                s.a.g[mhd::populationIndex<i, alpha>(id)] = 1e-7f * (static_cast<int>(i) - 3) * (alpha + 1) * (x + 2 * y + 7 * z);
            });
        });
    });
    nodes([&](auto x, auto y, auto z) { mhd::advanceNode(x, y, z, s.a, s.b); });
    nodes([&](auto x, auto y, auto z)
    {
        const auto id = idx(x, y, z);
        constexpr_for<0, 7>([&](auto I)
        {
            constexpr label_t i = decltype(I)::value;
            auto wrap = [](int coordinate, int c, int n) { return 1 + (coordinate - 1 - c + n) % n; };
            const auto source = idx(wrap(x, mhd::D3Q7::cx<i>(), NXPhysical), wrap(y, mhd::D3Q7::cy<i>(), NYPhysical), wrap(z, mhd::D3Q7::cz<i>(), NZPhysical));
            constexpr_for<0, 3>([&](auto A)
            {
                constexpr label_t alpha = decltype(A)::value;
                near(s.b.g[mhd::populationIndex<i, alpha>(id)], mhd::magneticOms * s.a.g[mhd::populationIndex<i, alpha>(source)], 1e-10, "magnetic kinetic-mode streaming");
            });
        });
    });
    std::cout << "PASS magnetic kinetic modes and independent periodic pull indices\n";
}

void checkHydrodynamicLimit()
{
    Buffers s;
    MomentState reference;
    allocateDeviceMoments(reference);
    nodes([&](auto x, auto y, auto z)
    {
        const auto id = idx(x, y, z);
        const mhd::Vector u{0.02f * sinf(mhd::k * x), 0.01f * cosf(mhd::k * y), 0.005f * sinf(mhd::k * (x + y))};
        setEquilibrium(s.a, id, 1 + 0.001f * cosf(mhd::k * y), u, {0, 0, 0});
        for (label_t field = 4; field < 10; ++field)
            s.a.fluid.data[field * fieldStride + id] = 0.0001f * sinf(mhd::k * (x + field * y));
    });
    nodes([&](auto x, auto y, auto z) { lbm(x, y, z, s.a.fluid, reference); });
    s.advance();
    nodes([&](auto x, auto y, auto z)
    {
        const auto id = idx(x, y, z);
        for (label_t field = 0; field < 10; ++field)
            near(s.a.fluid.data[field * fieldStride + id], reference.data[field * fieldStride + id], 1e-8, "zero-B hydrodynamic limit");
        near(s.a.bx[id], 0, 0, "zero B remains zero");
        near(s.a.by[id], 0, 0, "zero B remains zero");
        near(s.a.bz[id], 0, 0, "zero B remains zero");
    });
    freeDeviceMoments(reference);
    std::cout << "PASS zero-field reduction to the existing MR-R3 update\n";
}

void checkDecay()
{
    Buffers s;
    constexpr real_t amplitude = 0.01f;
    nodes([&](auto x, auto y, auto z)
    {
        const real_t phase = mhd::k * (x - 1);
        setEquilibrium(s.a, idx(x, y, z), 1, {0, 0, 0}, {0, amplitude * cosf(phase), amplitude * sinf(phase)});
    });
    // Circular, transverse B(x) has uniform magnetic pressure and J parallel B.
    constexpr int steps = 100;
    for (int i = 0; i < steps; ++i) s.advance();
    double error2 = 0, exact2 = 0, maxSpeed = 0;
    const double decay = std::exp(-mhd::diffusivity * mhd::k * mhd::k * steps);
    nodes([&](auto x, auto y, auto z)
    {
        const auto id = idx(x, y, z);
        const double phase = mhd::k * (x - 1);
        const double exactY = amplitude * decay * std::cos(phase), exactZ = amplitude * decay * std::sin(phase);
        error2 += std::pow(s.a.by[id] - exactY, 2) + std::pow(s.a.bz[id] - exactZ, 2);
        exact2 += exactY * exactY + exactZ * exactZ;
        maxSpeed = std::max(maxSpeed, std::abs(static_cast<double>(s.a.fluid.ux[id])));
    });
    const double relative = std::sqrt(error2 / exact2);
    require(relative < 0.005, "resistive decay exceeds 0.5% relative L2 error");
    require(maxSpeed < 2e-5, "force-free field spuriously drives fluid");
    std::cout << "PASS resistive decay: relative L2=" << relative << ", max|ux|=" << maxSpeed << '\n';
}

void checkAlfven()
{
    Buffers s;
    constexpr real_t background = 0.05f, amplitude = 0.001f;
    require(std::abs(mhd::viscosity - mhd::diffusivity) < 1e-8, "Alfven test requires nu=eta");
    nodes([&](auto x, auto y, auto z)
    {
        const real_t wave = amplitude * sinf(mhd::k * (x - 1));
        setEquilibrium(s.a, idx(x, y, z), 1, {0, -wave, 0}, {background, wave, 0});
    });
    constexpr int steps = 80;
    for (int i = 0; i < steps; ++i) s.advance();
    double error2 = 0, exact2 = 0;
    const double decay = std::exp(-mhd::diffusivity * mhd::k * mhd::k * steps);
    nodes([&](auto x, auto y, auto z)
    {
        const auto id = idx(x, y, z);
        // Right-traveling wave: u_y=-B_y for rho0=mu0=1; v_A=B_x.
        const double exact = amplitude * decay * std::sin(mhd::k * ((x - 1) - background * steps));
        error2 += std::pow(s.a.by[id] - exact, 2) + std::pow(s.a.fluid.uy[id] + exact, 2);
        exact2 += 2 * exact * exact;
    });
    const double relative = std::sqrt(error2 / exact2);
    require(relative < 0.03, "Alfven wave exceeds 3% relative L2 error");
    std::cout << "PASS coupled Alfven wave: relative L2=" << relative << '\n';
}

void checkOrszagTang(const char *outputDirectory)
{
    Buffers s;
    nodes([&](auto x, auto y, auto z) { mhd::initializeNode(x, y, z, s.a); });
    mhd::Snapshot h;
    mhd::copySnapshot(h, s.a);
    const auto initial = mhd::statistics(h);
    // The pressure-balanced density contributes an O(U0^2 B0^2 / cs2) correction.
    const double expectedK = 0.5 * rho0 * mhd::velocity * mhd::velocity
        - std::pow(mhd::velocity * mhd::magneticAmplitude, 2) / (32 * cs2);
    near(initial.kinetic, expectedK, 1e-10, "initial OT kinetic energy");
    near(initial.magnetic, 0.5 * mhd::magneticAmplitude * mhd::magneticAmplitude, 1e-10, "initial OT magnetic energy");
    near(initial.enstrophy, 0.5 * mhd::velocity * mhd::velocity * std::pow(std::sin(mhd::k), 2), 1e-10, "initial OT finite-difference enstrophy");
    near(initial.current2, 0.5 * mhd::magneticAmplitude * mhd::magneticAmplitude * (std::pow(std::sin(mhd::k), 2) + std::pow(std::sin(2 * mhd::k), 2)), 1e-10, "initial OT current");
    near(initial.divMax, 0, 0, "initial OT divergence");

    const std::filesystem::path dir(outputDirectory);
    std::filesystem::create_directories(dir);
    mhd::writeMetadata(dir);
    mhd::writeVTI(dir, h, 0);
    std::ofstream csv(dir / "statistics.csv");
    mhd::writeStatisticsHeader(csv); mhd::writeStatistics(csv, 0, initial);
    constexpr int steps = 100;
    for (int i = 0; i < steps; ++i) s.advance();
    mhd::copySnapshot(h, s.a);
    const auto final = mhd::statistics(h);
    const double massError = std::abs(final.mass / initial.mass - 1);
    require(massError < 3e-5, "OT mass conservation");
    require(final.kinetic + final.magnetic < initial.kinetic + initial.magnetic, "resistive viscous OT must lose total energy");
    for (const double b : final.meanB) near(b, 0, 2e-7, "OT mean magnetic flux");
    require(final.divRms * mhd::length / std::sqrt(2 * final.magnetic) < 0.02, "OT normalized centered divergence");
    mhd::writeStatistics(csv, steps, final);
    mhd::writeVTI(dir, h, steps);
    mhd::writePVD(dir, {0, steps});
    std::cout << "PASS OT evolution: relative mass error=" << massError << ", divB_rms=" << final.divRms
              << ", E_final/E_initial=" << (final.kinetic + final.magnetic) / (initial.kinetic + initial.magnetic) << '\n';
}
}

int main(int argc, char **argv)
{
    try
    {
        require(argc == 2, "Pass a temporary output directory.");
        checkMoments(); checkUniform(); checkMagneticStreaming(); checkHydrodynamicLimit(); checkDecay(); checkAlfven(); checkOrszagTang(argv[1]);
        std::cout << "All CPU numerical checks passed. CUDA compilation/execution is a separate check.\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

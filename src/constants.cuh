#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#include <cstddef>
#include <cstdint>
#include "utilities/types.cuh"

// ============================================================
// SELECT THE SIMULATION HERE (change only this line)
// ============================================================

enum class SimulationCase : std::uint8_t
{
    TaylorGreen,
    Jet,
    Channel
};

inline constexpr SimulationCase simulationCase = SimulationCase::Channel;

// ============================================================
// CUDA block dimensions
// ============================================================

inline constexpr label_t blockX = static_cast<label_t>(32);
inline constexpr label_t blockY = static_cast<label_t>(4);
inline constexpr label_t blockZ = static_cast<label_t>(4);

// ============================================================
// Independent grid settings (stored nodes)
//
// Taylor-Green and jet: one ghost plane on each side of x,y,z.
// Channel: periodic ghost planes in x,z; walls at y=0 and y=NY-1.
// ============================================================

inline constexpr label_t tgvGridNX = static_cast<label_t>(130);
inline constexpr label_t tgvGridNY = static_cast<label_t>(130);
inline constexpr label_t tgvGridNZ = static_cast<label_t>(130);

inline constexpr label_t jetGridNX = static_cast<label_t>(130);
inline constexpr label_t jetGridNY = static_cast<label_t>(402);
inline constexpr label_t jetGridNZ = static_cast<label_t>(130);

inline constexpr label_t channelGridNX = static_cast<label_t>(258);
inline constexpr label_t channelGridNY = static_cast<label_t>(129);
inline constexpr label_t channelGridNZ = static_cast<label_t>(130);

// ============================================================
// Active grid -- do not edit these expressions
// ============================================================

inline constexpr label_t NX =
    simulationCase == SimulationCase::TaylorGreen ? tgvGridNX :
    simulationCase == SimulationCase::Jet ? jetGridNX : channelGridNX;

inline constexpr label_t NY =
    simulationCase == SimulationCase::TaylorGreen ? tgvGridNY :
    simulationCase == SimulationCase::Jet ? jetGridNY : channelGridNY;

inline constexpr label_t NZ =
    simulationCase == SimulationCase::TaylorGreen ? tgvGridNZ :
    simulationCase == SimulationCase::Jet ? jetGridNZ : channelGridNZ;

static_assert(NX > 2 && NY > 2 && NZ > 2);

inline constexpr label_t NXPhysical = NX - static_cast<label_t>(2);
inline constexpr label_t NYPhysical = NY - static_cast<label_t>(2);
inline constexpr label_t NZPhysical = NZ - static_cast<label_t>(2);

inline constexpr std::size_t Nphysical =
    static_cast<std::size_t>(NXPhysical) * NYPhysical * NZPhysical;

inline constexpr label_t NXNY = NX * NY;
inline constexpr label_t Ncells = NX * NY * NZ;

static_assert(simulationCase != SimulationCase::TaylorGreen ||
              (NXPhysical == NYPhysical && NYPhysical == NZPhysical),
              "This TGV validation assumes a cubic physical domain");

// ============================================================
// Independent simulation lengths and VTI output frequencies
// All must fit label_t (uint32_t).
// ============================================================

inline constexpr label_t tgvSteps = static_cast<label_t>(9000);
inline constexpr label_t jetSteps = static_cast<label_t>(120000);
inline constexpr label_t channelSteps = static_cast<label_t>(200000);

inline constexpr label_t tgvOutputEvery = static_cast<label_t>(1000);
inline constexpr label_t jetOutputEvery = static_cast<label_t>(10000);
inline constexpr label_t channelOutputEvery = static_cast<label_t>(10000);

inline constexpr label_t Steps =
    simulationCase == SimulationCase::TaylorGreen ? tgvSteps :
    simulationCase == SimulationCase::Jet ? jetSteps : channelSteps;

inline constexpr label_t output =
    simulationCase == SimulationCase::TaylorGreen ? tgvOutputEvery :
    simulationCase == SimulationCase::Jet ? jetOutputEvery : channelOutputEvery;

// Used by Taylor-Green scalar-statistics output.
inline constexpr label_t statisticsStart = static_cast<label_t>(0);
inline constexpr label_t statisticsEvery = static_cast<label_t>(2);

static_assert(statisticsEvery > 0 && output > 0);

// ============================================================
// Velocity set, memory layout, and SoA buffer sizes
// ============================================================

inline constexpr label_t Q = static_cast<label_t>(27);
inline constexpr label_t Nmoments = static_cast<label_t>(10);
inline constexpr label_t NoutputFields = static_cast<label_t>(4);

inline constexpr std::size_t fieldAlignment = static_cast<std::size_t>(128);
inline constexpr std::size_t fieldAlignmentElements = fieldAlignment / sizeof(real_t);

inline constexpr std::size_t fieldStride =
    ((static_cast<std::size_t>(Ncells) + fieldAlignmentElements - 1) /
     fieldAlignmentElements) * fieldAlignmentElements;

inline constexpr std::size_t bytesField = fieldStride * sizeof(real_t);
inline constexpr std::size_t bytesMoments =
    static_cast<std::size_t>(Nmoments) * bytesField;
inline constexpr std::size_t bytesMomentsAB =
    static_cast<std::size_t>(2) * bytesMoments;
inline constexpr std::size_t bytesOutput =
    static_cast<std::size_t>(NoutputFields) * bytesField;

// ============================================================
// Mathematical and lattice constants
// ============================================================

inline constexpr real_t pi = static_cast<real_t>(3.14159265358979323846);
inline constexpr real_t cs2 = static_cast<real_t>(1.0 / 3.0);
inline constexpr real_t cs4 = cs2 * cs2;
inline constexpr real_t cs6 = cs4 * cs2;

inline constexpr real_t inv_cs2 = static_cast<real_t>(1) / cs2;
inline constexpr real_t inv_cs4 = static_cast<real_t>(1) / cs4;
inline constexpr real_t inv_2cs2 = static_cast<real_t>(1) / (static_cast<real_t>(2) * cs2);
inline constexpr real_t inv_2cs4 = static_cast<real_t>(1) / (static_cast<real_t>(2) * cs4);
inline constexpr real_t inv_6cs6 = static_cast<real_t>(1) / (static_cast<real_t>(6) * cs6);
inline constexpr real_t inv_2cs6 = static_cast<real_t>(1) / (static_cast<real_t>(2) * cs6);

inline constexpr real_t rho0 = static_cast<real_t>(1);

// ============================================================
// Taylor-Green vortex parameters
// ============================================================

inline constexpr label_t tgv_NX = NXPhysical;
inline constexpr label_t tgv_NY = NYPhysical;
inline constexpr label_t tgv_NZ = NZPhysical;

inline constexpr real_t tgv_U0 = static_cast<real_t>(0.05);
inline constexpr real_t tgv_kx = static_cast<real_t>(2) * pi / static_cast<real_t>(tgv_NX);
inline constexpr real_t tgv_ky = static_cast<real_t>(2) * pi / static_cast<real_t>(tgv_NY);
inline constexpr real_t tgv_kz = static_cast<real_t>(2) * pi / static_cast<real_t>(tgv_NZ);
inline constexpr real_t tgv_Re = static_cast<real_t>(1600);
inline constexpr real_t tgv_L = static_cast<real_t>(1) / tgv_kx;
inline constexpr real_t tgv_nu = tgv_U0 * tgv_L / tgv_Re;

// ============================================================
// Jet parameters
// ============================================================

inline constexpr real_t jet_radius = static_cast<real_t>(12.0);
inline constexpr real_t jet_diameter = static_cast<real_t>(2) * jet_radius;

inline constexpr real_t jet_x0 = static_cast<real_t>(NX - 1) / static_cast<real_t>(2);
inline constexpr real_t jet_z0 = static_cast<real_t>(NZ - 1) / static_cast<real_t>(2);

inline constexpr real_t jet_velocity = static_cast<real_t>(0.05);
inline constexpr real_t jet_Re = static_cast<real_t>(5000);
inline constexpr real_t jet_nu = jet_diameter * jet_velocity / jet_Re;

inline constexpr real_t jetStatsStartTStar = static_cast<real_t>(100.0);
inline constexpr label_t jetStatsSampleEvery = static_cast<label_t>(20);

// ============================================================
// Channel-flow parameters
// ============================================================

inline constexpr real_t channel_ReTau = static_cast<real_t>(180);
inline constexpr real_t channel_halfHeight =
    static_cast<real_t>(NY - 1) / static_cast<real_t>(2);
inline constexpr real_t channel_height = static_cast<real_t>(2) * channel_halfHeight;
inline constexpr real_t channel_uTau = static_cast<real_t>(0.01);
inline constexpr real_t channel_g = channel_uTau * channel_uTau / channel_halfHeight;
inline constexpr real_t channel_nu = channel_uTau * channel_halfHeight / channel_ReTau;

inline constexpr real_t channel_kappa = static_cast<real_t>(0.41);
inline constexpr real_t channel_B = static_cast<real_t>(5);
inline constexpr real_t channel_yPlusTransition = static_cast<real_t>(11);
inline constexpr real_t channel_perturbation =
    static_cast<real_t>(0.5) * channel_uTau;

inline constexpr real_t channel_kx =
    static_cast<real_t>(2) * pi / static_cast<real_t>(NX - 2);
inline constexpr real_t channel_kz =
    static_cast<real_t>(2) * pi / static_cast<real_t>(NZ - 2);

inline constexpr label_t channelStatsStartStep = static_cast<label_t>(100000);
inline constexpr label_t channelStatsSampleEvery = static_cast<label_t>(400);

// ============================================================
// Active viscosity, relaxation parameters
// ============================================================

inline constexpr real_t nu =
    simulationCase == SimulationCase::TaylorGreen ? tgv_nu :
    simulationCase == SimulationCase::Jet ? jet_nu : channel_nu;

inline constexpr real_t tau = static_cast<real_t>(0.5) + nu / cs2;
inline constexpr real_t omega = static_cast<real_t>(1) / tau;
inline constexpr real_t oms = static_cast<real_t>(1) - omega;

#endif // CONSTANTS_CUH

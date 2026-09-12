#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#include <cstddef>
#include "utilities/types.cuh"

// ============================================================
// CUDA block dimensions
// ============================================================

inline constexpr label_t blockX = static_cast<label_t>(32);
inline constexpr label_t blockY = static_cast<label_t>(4);
inline constexpr label_t blockZ = static_cast<label_t>(4);

// ============================================================
// Grid
// ============================================================

inline constexpr label_t NX = static_cast<label_t>(258);
inline constexpr label_t NY = static_cast<label_t>(258);
inline constexpr label_t NZ = static_cast<label_t>(258);

static_assert(NX > static_cast<label_t>(2) && NY > static_cast<label_t>(2) && NZ > static_cast<label_t>(2));

inline constexpr label_t NXPhysical = NX - static_cast<label_t>(2);
inline constexpr label_t NYPhysical = NY - static_cast<label_t>(2);
inline constexpr label_t NZPhysical = NZ - static_cast<label_t>(2);

inline constexpr std::size_t Nphysical = static_cast<std::size_t>(NXPhysical) * static_cast<std::size_t>(NYPhysical) * static_cast<std::size_t>(NZPhysical);

inline constexpr label_t NXNY = NX * NY;
inline constexpr label_t Ncells = NX * NY * NZ;

// ============================================================
// Simulation control
// ============================================================

inline constexpr label_t Steps = static_cast<label_t>(200000);
inline constexpr label_t output = static_cast<label_t>(1000);

inline constexpr label_t statisticsStart = static_cast<label_t>(0);
inline constexpr label_t statisticsEvery = static_cast<label_t>(2);

static_assert(statisticsEvery > static_cast<label_t>(0));

// ============================================================
// Velocity set D3Q27
// ============================================================

inline constexpr label_t Q = static_cast<label_t>(27);

// ============================================================
// Moment representation
// ============================================================

inline constexpr label_t Nmoments = static_cast<label_t>(10);

inline constexpr label_t NoutputFields = static_cast<label_t>(4);

// ============================================================
// Memory sizes
// ============================================================

inline constexpr std::size_t fieldAlignment = static_cast<std::size_t>(128);
inline constexpr std::size_t fieldAlignmentElements = fieldAlignment / sizeof(real_t);

inline constexpr std::size_t fieldStride = ((static_cast<std::size_t>(Ncells) + fieldAlignmentElements - 1) / fieldAlignmentElements) * fieldAlignmentElements;

inline constexpr std::size_t bytesField = fieldStride * sizeof(real_t);
inline constexpr std::size_t bytesMoments = static_cast<std::size_t>(Nmoments) * bytesField;
inline constexpr std::size_t bytesMomentsAB = static_cast<std::size_t>(2) * bytesMoments;

inline constexpr std::size_t bytesOutput = static_cast<std::size_t>(NoutputFields) * bytesField;

// ============================================================
// Mathematical constants
// ============================================================

inline constexpr real_t pi = static_cast<real_t>(3.14159265358979323846);

// ============================================================
// Lattice constants
// ============================================================

inline constexpr real_t cs2 = static_cast<real_t>(1.0 / 3.0);
inline constexpr real_t cs4 = cs2 * cs2;
inline constexpr real_t cs6 = cs4 * cs2;

inline constexpr real_t inv_cs2 = static_cast<real_t>(1) / cs2;
inline constexpr real_t inv_cs4 = static_cast<real_t>(1) / cs4;
inline constexpr real_t inv_2cs2 = static_cast<real_t>(1) / (static_cast<real_t>(2) * cs2);
inline constexpr real_t inv_2cs4 = static_cast<real_t>(1) / (static_cast<real_t>(2) * cs4);
inline constexpr real_t inv_6cs6 = static_cast<real_t>(1) / (static_cast<real_t>(6) * cs6);
inline constexpr real_t inv_2cs6 = static_cast<real_t>(1) / (static_cast<real_t>(2) * cs6);

// ============================================================
// General fluid parameters
// ============================================================

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

// ============================================================
// Channel-flow parameters
// ============================================================

inline constexpr real_t channel_ReTau = static_cast<real_t>(180);

inline constexpr real_t channel_halfHeight = static_cast<real_t>(NY - 1) / static_cast<real_t>(2);
inline constexpr real_t channel_height = static_cast<real_t>(2) * channel_halfHeight;

inline constexpr real_t channel_uTau = static_cast<real_t>(0.01);

inline constexpr real_t channel_g = channel_uTau * channel_uTau / channel_halfHeight;

inline constexpr real_t channel_nu = channel_uTau * channel_halfHeight / channel_ReTau;

// Approximate initial turbulent mean profile
inline constexpr real_t channel_kappa = static_cast<real_t>(0.41);
inline constexpr real_t channel_B = static_cast<real_t>(5);
inline constexpr real_t channel_yPlusTransition = static_cast<real_t>(11);

// Initial perturbation
inline constexpr real_t channel_perturbation = static_cast<real_t>(0.5) * channel_uTau;

// Periodic wavelengths
inline constexpr real_t channel_kx = static_cast<real_t>(2) * pi / static_cast<real_t>(NX - 2);
inline constexpr real_t channel_kz = static_cast<real_t>(2) * pi / static_cast<real_t>(NZ - 2);

// ============================================================
// Active fluid parameters
// ============================================================

// Change this manually depending on the case:
//
// Jet:
// inline constexpr real_t nu = jet_nu;
//
// Taylor-Green:
// inline constexpr real_t nu = tgv_nu;
//
// Channel:
// inline constexpr real_t nu = channel_nu;

inline constexpr real_t nu = tgv_nu;

inline constexpr real_t tau = static_cast<real_t>(0.5) + nu / cs2;
inline constexpr real_t omega = static_cast<real_t>(1) / tau;
inline constexpr real_t oms = static_cast<real_t>(1) - omega;

#endif

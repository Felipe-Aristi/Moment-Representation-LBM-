#ifndef MHD_CONFIG_CUH
#define MHD_CONFIG_CUH

#include "../../src/utilities/types.cuh"
#include <limits>

// All numerical choices are compile-time. Override with -DMHD_NX=..., etc.
#ifndef MHD_NX
#define MHD_NX 128
#endif
#ifndef MHD_NY
#define MHD_NY MHD_NX
#endif
#ifndef MHD_NZ
#define MHD_NZ 1
#endif
#ifndef MHD_STEPS
#define MHD_STEPS 10000
#endif
#ifndef MHD_RE
#define MHD_RE 100.0
#endif
#ifndef MHD_RM
#define MHD_RM 100.0
#endif

namespace mhd
{
inline constexpr label_t nx = MHD_NX;
inline constexpr label_t ny = MHD_NY;
inline constexpr label_t nz = MHD_NZ;
inline constexpr label_t steps = MHD_STEPS;
inline constexpr label_t outputEvery = 500;
inline constexpr label_t sampleEvery = 50;
inline constexpr real_t twoPi = static_cast<real_t>(6.28318530717958647693);
inline constexpr real_t k = twoPi / static_cast<real_t>(nx);
inline constexpr real_t length = static_cast<real_t>(1) / k;
inline constexpr real_t velocity = static_cast<real_t>(0.02);
inline constexpr real_t magneticAmplitude = static_cast<real_t>(0.02);
inline constexpr real_t reynolds = static_cast<real_t>(MHD_RE);
inline constexpr real_t magneticReynolds = static_cast<real_t>(MHD_RM);
inline constexpr real_t viscosity = velocity * length / reynolds;
inline constexpr real_t diffusivity = velocity * length / magneticReynolds;
inline constexpr real_t magneticCs2 = static_cast<real_t>(0.25);
inline constexpr real_t magneticTau = static_cast<real_t>(0.5) + diffusivity / magneticCs2;
inline constexpr real_t magneticOmega = static_cast<real_t>(1) / magneticTau;
inline constexpr real_t magneticOms = static_cast<real_t>(1) - magneticOmega;

// B is expressed in units with magnetic permeability mu0 = 1.
static_assert(MHD_NX >= 8 && MHD_NY == MHD_NX && MHD_NZ >= 1, "Orszag-Tang requires a square x-y domain and at least one z plane.");
static_assert(MHD_STEPS >= 0 && reynolds > 0 && magneticReynolds > 0);
static_assert(static_cast<unsigned long long>(MHD_NX + 2) * (MHD_NY + 2) * (MHD_NZ + 2) <= std::numeric_limits<label_t>::max(), "Stored grid exceeds label_t indexing capacity.");
static_assert(static_cast<unsigned long long>(MHD_STEPS) < std::numeric_limits<label_t>::max(), "Step count exceeds loop indexing capacity.");
static_assert(magneticTau > static_cast<real_t>(0.5));
static_assert(outputEvery > 0 && sampleEvery > 0);
}
#endif

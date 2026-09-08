#ifndef MHD_CONSTANTSMHD_CUH
#define MHD_CONSTANTSMHD_CUH

#include "../../src/constants.cuh"
#include "../../stencils/d3q7.cuh"

// ============================================================
// Electromagnetic constants
// ============================================================

inline constexpr real_t mhd_mu0 = static_cast<real_t>(1);
inline constexpr real_t mhd_inv_mu0 = static_cast<real_t>(1) / mhd_mu0;

// ============================================================
// Magnetic lattice constants
// ============================================================

inline constexpr real_t mhd_cs2 = d3q7::cs2;
inline constexpr real_t mhd_inv_cs2 = d3q7::inv_cs2;

// ============================================================
// Magnetic Prandtl number
//
// Pm = nu / eta
// ============================================================

inline constexpr real_t mhd_Pm = static_cast<real_t>(1);

// ============================================================
// Magnetic diffusivity and relaxation
//
// eta = cs_m^2 (tauM - 1/2)
// ============================================================

inline constexpr real_t mhd_eta = nu / mhd_Pm;
inline constexpr real_t tauM = static_cast<real_t>(0.5) + mhd_eta / mhd_cs2;
inline constexpr real_t omegaM = static_cast<real_t>(1) / tauM;
inline constexpr real_t omsM = static_cast<real_t>(1) - omegaM;

// ============================================================
// Orszag-Tang vortex
// ============================================================

inline constexpr label_t ot_NX = NXPhysical;
inline constexpr label_t ot_NY = NYPhysical;

inline constexpr real_t ot_U0 = static_cast<real_t>(0.05);
inline constexpr real_t ot_B0 = static_cast<real_t>(0.05);

inline constexpr real_t ot_kx = static_cast<real_t>(2) * pi / static_cast<real_t>(ot_NX);
inline constexpr real_t ot_ky = static_cast<real_t>(2) * pi / static_cast<real_t>(ot_NY);

#endif
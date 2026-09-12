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
// Orszag-Tang vortex geometry
// ============================================================

inline constexpr label_t ot_NX = NXPhysical;
inline constexpr label_t ot_NY = NYPhysical;

inline constexpr real_t ot_kx = static_cast<real_t>(2) * pi / static_cast<real_t>(ot_NX);
inline constexpr real_t ot_ky = static_cast<real_t>(2) * pi / static_cast<real_t>(ot_NY);

inline constexpr real_t ot_L = static_cast<real_t>(1) / ot_kx;

// ============================================================
// Orszag-Tang characteristic amplitudes
// ============================================================

inline constexpr real_t ot_U0 = static_cast<real_t>(0.05);
inline constexpr real_t ot_B0 = static_cast<real_t>(0.05);

// ============================================================
// Orszag-Tang dimensionless transport parameters
//
// Re = U0 L / nu
// Rm = U0 L / eta
// Pm = nu / eta = Rm / Re
// ============================================================

inline constexpr real_t ot_Re = static_cast<real_t>(100);
inline constexpr real_t ot_Pm = static_cast<real_t>(1);
inline constexpr real_t ot_Rm = ot_Re * ot_Pm;

// ============================================================
// Fluid viscosity and relaxation
//
// nu = U0 L / Re
// nu = cs_f^2 (tauF - 1/2)
// ============================================================

inline constexpr real_t mhd_nu = ot_U0 * ot_L / ot_Re;

inline constexpr real_t mhd_tauF = static_cast<real_t>(0.5) + mhd_nu / cs2;
inline constexpr real_t mhd_omegaF = static_cast<real_t>(1) / mhd_tauF;
inline constexpr real_t mhd_omsF = static_cast<real_t>(1) - mhd_omegaF;

// ============================================================
// Magnetic diffusivity and relaxation
//
// eta = nu / Pm
// eta = cs_m^2 (tauM - 1/2)
// ============================================================

inline constexpr real_t mhd_eta = mhd_nu / ot_Pm;

inline constexpr real_t mhd_tauM = static_cast<real_t>(0.5) + mhd_eta / mhd_cs2;
inline constexpr real_t mhd_omegaM = static_cast<real_t>(1) / mhd_tauM;
inline constexpr real_t mhd_omsM = static_cast<real_t>(1) - mhd_omegaM;



#endif
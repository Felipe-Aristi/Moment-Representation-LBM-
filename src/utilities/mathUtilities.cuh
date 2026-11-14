#ifndef MATH_UTILITIES_CUH
#define MATH_UTILITIES_CUH

#include <cmath>
#include "types.cuh"
#include "../constants.cuh"

// =======================================================
// Jet mask
// =======================================================

__device__ __forceinline__ bool isJet(const label_t x, const label_t z) noexcept
{
    const real_t dx = static_cast<real_t>(x) - jet_x0;
    const real_t dz = static_cast<real_t>(z) - jet_z0;

    return (dx * dx + dz * dz) <= (jet_radius * jet_radius) ? 1 : 0;
}

// ============================================================
// Convective update
// phi_B^(n+1) = phi_B^n - uc (phi_B^n - phi_F^n)
// ============================================================

__device__ __forceinline__ real_t convectiveOutlet(const real_t phiB, const real_t phiF, const real_t uc) noexcept
{
    return phiB - uc * (phiB - phiF);
}


 // ============================================================
 // Jet outlet sponge layer
 //
 // nu_eff(y) = nu * (1 + K*s^3)
 // tau_eff(y) = 0.5 + nu_eff(y)/cs2
 // ============================================================

__device__ __forceinline__ real_t jetSpongeOmega(const label_t y) noexcept
{
    const label_t yc = y > jetSpongeYEnd ? jetSpongeYEnd : y;

    if (yc < jetSpongeYStart)
        return omega;

    const real_t s = static_cast<real_t>(yc - jetSpongeYStart) / static_cast<real_t>(jetSpongeYEnd - jetSpongeYStart);

    const real_t s3 = s * s * s;

    const real_t nuEff = jet_nu * (static_cast<real_t>(1) + jetSpongeGain * s3);

    const real_t tauEff = static_cast<real_t>(0.5) + nuEff / cs2;

    return static_cast<real_t>(1) / tauEff;
}



 // ============================================================
 // Deterministic pseudorandom number generator
 // ============================================================

__device__ __forceinline__ unsigned int jetNoiseHash(unsigned int value) noexcept
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;

    return value;
}

// ============================================================
// Uniform pseudorandom value in [-1, 1]
// ============================================================

__device__ __forceinline__ real_t jetRandomValue(
    const unsigned int interval,
    const unsigned int seed) noexcept
{
    const unsigned int value = jetNoiseHash(interval ^ seed);

    return static_cast<real_t>(2) *
               static_cast<real_t>(value >> 8) /
               static_cast<real_t>(16777215) -
           static_cast<real_t>(1);
}

// ============================================================
// Temporally correlated noise
// ============================================================

__device__ __forceinline__ real_t jetCorrelatedNoise(
    const label_t step,
    const label_t interval,
    const unsigned int seed) noexcept
{
    const unsigned int k = step / interval;

    const real_t t =
        static_cast<real_t>(step % interval) /
        static_cast<real_t>(interval);

    const real_t smooth = t * t * (static_cast<real_t>(3) - static_cast<real_t>(2) * t);

    const real_t a = jetRandomValue(k, seed);
    const real_t b = jetRandomValue(k + 1u, seed);

    return a + (b - a) * smooth;
}

// ============================================================
// Perturbed axial jet inlet velocity
//
// m = 1: helical-type azimuthal disturbance
// m = 2: second azimuthal disturbance
// ============================================================

__device__ __forceinline__ real_t jetInletVelocity(
    const label_t x,
    const label_t z,
    const label_t step) noexcept
{
    if (!isJet(x, z))
        return static_cast<real_t>(0);

    const real_t dx =
        (static_cast<real_t>(x) - jet_x0) / jet_radius;

    const real_t dz =
        (static_cast<real_t>(z) - jet_z0) / jet_radius;

    const real_t r2 = dx * dx + dz * dz;
    const real_t r4 = r2 * r2;

    // Smooth envelope, strongest away from the jet axis.
    // Vanishes at the circular nozzle edge.
    const real_t envelope = static_cast<real_t>(1) - r4 * r4;

    // Slowly varying random coefficients.
    const real_t a = jetCorrelatedNoise(step, jetNoiseCorrelationSteps, 0x243f6a88u);
    const real_t b = jetCorrelatedNoise(step, jetNoiseCorrelationSteps, 0x85a308d3u);

    // Faster, lower-amplitude secondary modes.
    const real_t c = jetCorrelatedNoise(step, jetNoiseCorrelationSteps / 2, 0x13198a2eu);
    const real_t d = jetCorrelatedNoise(step, jetNoiseCorrelationSteps / 2, 0x03707344u);

    const real_t mode1 = dx * a + dz * b;

    const real_t mode2 =
        (dx * dx - dz * dz) * c +
        static_cast<real_t>(2) * dx * dz * d;

    const real_t disturbance =
        jetNoiseAmplitude * envelope *
        (mode1 + static_cast<real_t>(0.35) * mode2);

    return jet_velocity * (static_cast<real_t>(1) + disturbance);
}






#endif
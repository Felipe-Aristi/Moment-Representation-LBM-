#include "kernels.cuh"

#include "lbm.cuh"
#include "lbmf.cuh"

#include "utilities/bounds.cuh"
#include "utilities/indexing.cuh"
#include "utilities/constexprFor.cuh"

#include "../cases/taylorgreen/initialization.cuh"

#include "../cases/jet/initialization.cuh"
#include "../cases/jet/boundary_conditions.cuh"

#include "../cases/channelflow/initialization.cuh"
#include "../cases/channelflow/boundary_conditions.cuh"

// ============================================================
// Taylor-Green initialization kernel
// ============================================================

__global__ void initializeTaylorGreenKernel(MomentState state)
{
    const label_t x = static_cast<label_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const label_t y = static_cast<label_t>(blockIdx.y * blockDim.y + threadIdx.y);
    const label_t z = static_cast<label_t>(blockIdx.z * blockDim.z + threadIdx.z);

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    initializeTaylorGreen(x, y, z, state);
}

// ============================================================
// Jet bulk initialization
// ============================================================

__global__ void initializeJetBulkKernel(MomentState state)
{
    const label_t x = static_cast<label_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const label_t y = static_cast<label_t>(blockIdx.y * blockDim.y + threadIdx.y);
    const label_t z = static_cast<label_t>(blockIdx.z * blockDim.z + threadIdx.z);

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    initializeJetBulk(x, y, z, state);
}

// ============================================================
// Jet inlet initialization
// ============================================================

__global__ void initializeJetInletKernel(MomentState state)
{
    const label_t x = static_cast<label_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const label_t z = static_cast<label_t>(blockIdx.y * blockDim.y + threadIdx.y);

    if (inlet_outlet_interior(x, z))
    {
        return;
    }

    initializeJetInlet(x, z, state);
}

// ============================================================
// Jet boundary conditions
// ============================================================

__global__ void jetBoundaryConditionsKernel(const __grid_constant__ MomentState A, const __grid_constant__ MomentState B)
{
    const label_t x = static_cast<label_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const label_t z = static_cast<label_t>(blockIdx.y * blockDim.y + threadIdx.y);

    if (inlet_outlet_interior(x, z))
    {
        return;
    }

    jetBoundaryConditions(x, z, A, B);
}

// ============================================================
// Channel-flow initialization
// ============================================================

__global__ void initializeChannelFlowKernel(MomentState state)
{
    const label_t x = static_cast<label_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const label_t y = static_cast<label_t>(blockIdx.y * blockDim.y + threadIdx.y);
    const label_t z = static_cast<label_t>(blockIdx.z * blockDim.z + threadIdx.z);

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    initializeChannelFlow(x, y, z, state);
}

// ============================================================
// Channel-flow boundary conditions
// ============================================================

__global__ void channelBoundaryConditionsKernel(const __grid_constant__ MomentState A,
                                                const __grid_constant__ MomentState B)
{
    const label_t x = static_cast<label_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const label_t z = static_cast<label_t>(blockIdx.y * blockDim.y + threadIdx.y);

    if (inlet_outlet_interior(x, z))
    {
        return;
    }

    channelBoundaryConditions(x, z, A, B);
}

// ============================================================
// MR-LBM kernel
// ============================================================

__global__ void lbmKernel(const __grid_constant__ MomentState A, const __grid_constant__ MomentState B)
{
    const label_t x = static_cast<label_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const label_t y = static_cast<label_t>(blockIdx.y * blockDim.y + threadIdx.y);
    const label_t z = static_cast<label_t>(blockIdx.z * blockDim.z + threadIdx.z);

    if (interior(x, y, z))
    {
        return;
    }

    lbm(x, y, z, A, B);
}

__global__ void lbmfKernel(const __grid_constant__ MomentState A, const __grid_constant__ MomentState B)
{
    const label_t x = static_cast<label_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const label_t y = static_cast<label_t>(blockIdx.y * blockDim.y + threadIdx.y);
    const label_t z = static_cast<label_t>(blockIdx.z * blockDim.z + threadIdx.z);

    if (interior(x, y, z))
    {
        return;
    }

    lbmf(x, y, z, A, B);
}

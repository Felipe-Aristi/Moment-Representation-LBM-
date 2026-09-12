#include "kernels.cuh"

#include "lbm.cuh"
#include "lbmf.cuh"

#include "statistics/kineticEnergy.cuh"
#include "statistics/enstrophy.cuh"
#include "statistics/reduction.cuh"
#include "statistics/statisticsMemory.cuh"

#include "utilities/bounds.cuh"
#include "utilities/indexing.cuh"
#include "utilities/constexprFor.cuh"

#include "../cases/taylorgreen/initialization.cuh"

#include "../cases/jet/initialization.cuh"
#include "../cases/jet/boundary_conditions.cuh"

#include "../cases/channelflow/initialization.cuh"
#include "../cases/channelflow/boundary_conditions.cuh"

#include "../cases/MHD/initialization.cuh"
#include "../cases/MHD/lbmmhd.cuh"

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
// Kinetic-energy partial sums
// ============================================================

__global__ void kineticEnergyPartialsKernel(const __grid_constant__ MomentState state, statistics_t *partials)
{
    const std::size_t blockStart = static_cast<std::size_t>(blockIdx.x) * statisticsNodesPerBlock;
    const std::size_t firstPhysicalId = blockStart + static_cast<std::size_t>(threadIdx.x);
    const std::size_t secondPhysicalId = firstPhysicalId + static_cast<std::size_t>(statisticsBlockSize);

    statistics_t localSum = static_cast<statistics_t>(0);

    if (firstPhysicalId < Nphysical)
    {
        const label_t id = physicalNodeIndex(firstPhysicalId);
        localSum += kineticEnergy(state.ux[id], state.uy[id], state.uz[id]);
    }

    if (secondPhysicalId < Nphysical)
    {
        const label_t id = physicalNodeIndex(secondPhysicalId);
        localSum += kineticEnergy(state.ux[id], state.uy[id], state.uz[id]);
    }

    const statistics_t blockSum = blockReduceSum(localSum);

    if (threadIdx.x == static_cast<unsigned int>(0))
    {
        partials[blockIdx.x] = blockSum;
    }
}

// ============================================================
// Enstrophy partial sums
// ============================================================

__global__ void enstrophyPartialsKernel(const __grid_constant__ MomentState state, statistics_t *partials)
{
    const std::size_t blockStart = static_cast<std::size_t>(blockIdx.x) * statisticsNodesPerBlock;
    const std::size_t firstPhysicalId = blockStart + static_cast<std::size_t>(threadIdx.x);
    const std::size_t secondPhysicalId = firstPhysicalId + static_cast<std::size_t>(statisticsBlockSize);

    statistics_t localSum = static_cast<statistics_t>(0);

    if (firstPhysicalId < Nphysical)
    {
        localSum += enstrophy(firstPhysicalId, state);
    }

    if (secondPhysicalId < Nphysical)
    {
        localSum += enstrophy(secondPhysicalId, state);
    }

    const statistics_t blockSum = blockReduceSum(localSum);

    if (threadIdx.x == static_cast<unsigned int>(0))
    {
        partials[blockIdx.x] = blockSum;
    }
}

// ============================================================
// Final statistics reduction
// ============================================================

__global__ void statisticsFinalReductionKernel(const statistics_t *partials, statistics_t *result)
{
    statistics_t localSum = static_cast<statistics_t>(0);

    for (std::size_t partialId = static_cast<std::size_t>(threadIdx.x); partialId < statisticsPartialCount; partialId += static_cast<std::size_t>(statisticsBlockSize))
    {
        localSum += partials[partialId];
    }

    const statistics_t sum = blockReduceSum(localSum);

    if (threadIdx.x == static_cast<unsigned int>(0))
    {
        *result = sum;
    }
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


// ============================================================
// Orszag-Tang initialization kernel
// ============================================================

__global__ void initializeOrszagTangKernel(MomentState fluid, MagneticState magnetic)
{
    const label_t x = static_cast<label_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const label_t y = static_cast<label_t>(blockIdx.y * blockDim.y + threadIdx.y);
    const label_t z = static_cast<label_t>(blockIdx.z * blockDim.z + threadIdx.z);

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    initializeOrszagTang(x, y, z, fluid, magnetic);
}

// ============================================================
// MR-MHD kernel
// ============================================================

__global__ void lbmmhdKernel(const __grid_constant__ MomentState fluidA, const __grid_constant__ MomentState fluidB,
                             const __grid_constant__ MagneticState magneticA, const __grid_constant__ MagneticState magneticB)
{
    const label_t x = static_cast<label_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const label_t y = static_cast<label_t>(blockIdx.y * blockDim.y + threadIdx.y);
    const label_t z = static_cast<label_t>(blockIdx.z * blockDim.z + threadIdx.z);

    if (interior(x, y, z))
    {
        return;
    }

    lbmmhd(x, y, z, fluidA, fluidB, magneticA, magneticB);
}
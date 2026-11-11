#include <cuda_runtime.h>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <utility>

#include "constants.cuh"
#include "memory.cuh"
#include "launch.cuh"

#include "statistics/sampling.cuh"
#include "statistics/statisticsMemory.cuh"

#include "utilities/cudaConfig.cuh"
#include "utilities/cudaUtilities.cuh"
#include "utilities/types.cuh"
#include "utilities/mlups_count.cuh"

#include "../cases/taylorgreen/statistics.cuh"
#include "../cases/jet/statistics.cuh"
#include "../cases/jet/energy_statistics.cuh"
#include "../cases/channelflow/statistics.cuh"
#include "../cases/channelflow/energy_statistics.cuh"

#include "../post/taylorgreen/vti.cuh"
#include "../post/jet/vti.cuh"
#include "../post/channelflow/vti.cuh"

// ============================================================
// Case-specific VTI output
// ============================================================

template <SimulationCase Case>
inline void writeSelectedVTI(Memory &memory, const MomentState &state,
                             const label_t step)
{
    copyDeviceToHost(memory.host, state);

    if constexpr (Case == SimulationCase::TaylorGreen)
    {
        writeTaylorGreenVTI(memory.host, step);
    }
    else if constexpr (Case == SimulationCase::Jet)
    {
        writeJetVTI(memory.host, step);
    }
    else
    {
        writeChannelFlowVTI(memory.host, step);
    }
}

// ============================================================
// Run one compile-time-selected simulation
// ============================================================

template <SimulationCase Case>
int runSimulation()
{
    constexpr int deviceID = 0;

    constexpr label_t jetEnergySampleEvery = static_cast<label_t>(2000);
    constexpr label_t channelEnergySampleEvery = static_cast<label_t>(200);
    constexpr label_t channelStatsWriteEvery = static_cast<label_t>(10000);

    static_assert(output > static_cast<label_t>(0));
    static_assert(statisticsEvery > static_cast<label_t>(0));
    static_assert(jetStatsSampleEvery > static_cast<label_t>(0));
    static_assert(channelStatsSampleEvery > static_cast<label_t>(0));

    CUDA_CHECK(cudaSetDevice(deviceID));
    const CudaConfig cfg = print_device_and_make_config(deviceID);

    Memory memory{};
    StatisticsMemory reduction{};
    MlupsCounter mlupsCounter{};
    bool mlupsInitialized = false;

    TaylorGreenStatisticsOutput tgvOutput{};
    JetRadialStatistics jetStatistics{};
    JetKineticEnergyHistory jetEnergy{};
    ChannelFlowStatistics channelStatistics{};
    ChannelKineticEnergyHistory channelEnergy{};

    int exitCode = EXIT_SUCCESS;

    try
    {
        allocateMemory(memory);
        allocateStatisticsMemory(reduction);

        // ====================================================
        // Initialize only the selected case
        // ====================================================

        if constexpr (Case == SimulationCase::TaylorGreen)
        {
            initializeTaylorGreenStatisticsOutput(tgvOutput);
            launchTaylorGreenInitialization(cfg, memory.d_A);
        }
        else if constexpr (Case == SimulationCase::Jet)
        {
            initializeJetRadialStatistics(jetStatistics);
            initializeJetKineticEnergyHistory(jetEnergy);
            launchJetInitialization(cfg, memory.d_A);
        }
        else
        {
            initializeChannelFlowStatistics(channelStatistics);
            initializeChannelKineticEnergyHistory(
                channelEnergy, channelStatistics.directory);
            launchChannelFlowInitialization(cfg, memory.d_A);
        }

        initializeMlupsCounter(mlupsCounter);
        mlupsInitialized = true;

        MomentState *A = &memory.d_A;
        MomentState *B = &memory.d_B;

        // ====================================================
        // Step-zero scalar statistics
        // ====================================================

        if constexpr (Case == SimulationCase::TaylorGreen)
        {
            const statistics_t energy =
                launchKineticEnergyStatistics(*A, reduction);
            const statistics_t enstrophy =
                launchEnstrophyStatistics(*A, reduction);

            writeTaylorGreenKineticEnergy(tgvOutput, 0, energy);
            writeTaylorGreenEnstrophy(tgvOutput, 0, enstrophy);
        }
        else if constexpr (Case == SimulationCase::Jet)
        {
            recordJetKineticEnergy(
                jetEnergy, 0, launchKineticEnergyStatistics(*A, reduction));
        }
        else
        {
            recordChannelKineticEnergy(
                channelEnergy, 0, launchKineticEnergyStatistics(*A, reduction));
        }

        writeSelectedVTI<Case>(memory, *A, 0);

        std::cout << "\n========================================\n";
        if constexpr (Case == SimulationCase::TaylorGreen)
            std::cout << "Taylor-Green vortex | Re = " << tgv_Re << '\n';
        else if constexpr (Case == SimulationCase::Jet)
            std::cout << "Turbulent jet | Re = " << jet_Re << '\n';
        else
            std::cout << "Turbulent channel | Re_tau = " << channel_ReTau << '\n';

        std::cout << "Physical grid: " << NXPhysical << " x "
                  << NYPhysical << " x " << NZPhysical << '\n';
        std::cout << "nu = " << nu << ", tau = " << tau << '\n';
        std::cout << "Steps = " << Steps << ", VTI interval = " << output << '\n';
        std::cout << "========================================\n\n";
        printMlupsHeader();

        // ====================================================
        // Time evolution
        // ====================================================

        for (label_t step = 0; step < Steps; ++step)
        {
            if (step % mlupsEvery == 0)
                startMlupsCounter(mlupsCounter);

            // ------------------------------------------------
            // Streaming and collision; force only for channel
            // ------------------------------------------------

            if constexpr (Case == SimulationCase::Channel)
            {
                launchLBMF(cfg, *A, *B);
                launchChannelBoundaryConditions(cfg, *A, *B);
            }
            else if constexpr (Case == SimulationCase::Jet)
            {
                launchLBM(cfg, *A, *B);
                launchJetBoundaryConditions(cfg, *A, *B);
            }
            else
            {
                launchLBM(cfg, *A, *B);
            }

            std::swap(A, B);
            const label_t completedStep = step + 1;

            // ------------------------------------------------
            // Taylor-Green: energy and enstrophy
            // ------------------------------------------------

            if constexpr (Case == SimulationCase::TaylorGreen)
            {
                if (statisticsSampleDue<statisticsStart, statisticsEvery>(completedStep))
                {
                    const statistics_t energy =
                        launchKineticEnergyStatistics(*A, reduction);
                    const statistics_t enstrophy =
                        launchEnstrophyStatistics(*A, reduction);

                    writeTaylorGreenKineticEnergy(tgvOutput, completedStep, energy);
                    writeTaylorGreenEnstrophy(tgvOutput, completedStep, enstrophy);
                }
            }

            // ------------------------------------------------
            // Jet: kinetic-energy history and radial profiles
            // ------------------------------------------------

            if constexpr (Case == SimulationCase::Jet)
            {
                if (completedStep % jetEnergySampleEvery == 0)
                {
                    recordJetKineticEnergy(
                        jetEnergy, completedStep,
                        launchKineticEnergyStatistics(*A, reduction));
                }

                if (jetRadialSampleDue(completedStep))
                {
                    if (jetStatistics.sampledSteps == 0)
                        std::cout << "Starting jet radial statistics at step "
                                  << completedStep << '\n';

                    launchJetRadialStatistics(cfg, *A, jetStatistics, completedStep);
                }

                if (completedStep % output == 0 && jetStatistics.sampledSteps > 0)
                    writeJetRadialStatistics(jetStatistics);
            }

            // ------------------------------------------------
            // Channel: kinetic-energy history and plane moments
            // ------------------------------------------------

            if constexpr (Case == SimulationCase::Channel)
            {
                if (completedStep % channelEnergySampleEvery == 0)
                {
                    recordChannelKineticEnergy(
                        channelEnergy, completedStep,
                        launchKineticEnergyStatistics(*A, reduction));
                }

                if (statisticsSampleDue<channelStatsStartStep,
                                        channelStatsSampleEvery>(completedStep))
                {
                    if (channelStatistics.sampleCount == 0)
                        std::cout << "Starting channel profiles at step "
                                  << completedStep << '\n';

                    launchChannelFlowStatistics(
                        *A, channelStatistics, completedStep);
                }

                if (completedStep % channelStatsWriteEvery == 0 &&
                    channelStatistics.sampleCount > 0)
                    writeChannelFlowStatistics(channelStatistics);
            }

            // ------------------------------------------------
            // Performance and visualization output
            // ------------------------------------------------

            if (completedStep % mlupsEvery == 0)
                printMlups(mlupsCounter, completedStep);

            if (completedStep % output == 0)
                writeSelectedVTI<Case>(memory, *A, completedStep);
        }

        // ====================================================
        // Final records (avoid duplicate step numbers)
        // ====================================================

        if constexpr (Case == SimulationCase::Jet)
        {
            if (Steps % jetEnergySampleEvery != 0)
                recordJetKineticEnergy(
                    jetEnergy, Steps,
                    launchKineticEnergyStatistics(*A, reduction));

            if (jetStatistics.sampledSteps > 0)
                writeJetRadialStatistics(jetStatistics);

            closeJetKineticEnergyHistory(jetEnergy);
        }
        else if constexpr (Case == SimulationCase::Channel)
        {
            if (Steps % channelEnergySampleEvery != 0)
                recordChannelKineticEnergy(
                    channelEnergy, Steps,
                    launchKineticEnergyStatistics(*A, reduction));

            if (channelStatistics.sampleCount > 0)
                writeChannelFlowStatistics(channelStatistics);

            closeChannelKineticEnergyHistory(channelEnergy);
        }
        else
        {
            closeTaylorGreenStatisticsOutput(tgvOutput);
        }

        if (Steps % output != 0)
            writeSelectedVTI<Case>(memory, *A, Steps);

        std::cout << "\nSimulation completed.\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << "Simulation error: " << error.what() << '\n';
        exitCode = EXIT_FAILURE;
    }

    // ========================================================
    // Cleanup
    // ========================================================

    CUDA_CHECK(cudaDeviceSynchronize());

    if (mlupsInitialized)
        destroyMlupsCounter(mlupsCounter);

    if constexpr (Case == SimulationCase::Jet)
        freeJetRadialStatistics(jetStatistics);
    else if constexpr (Case == SimulationCase::Channel)
        freeChannelFlowStatistics(channelStatistics);

    freeStatisticsMemory(reduction);
    freeMemory(memory);

    CUDA_CHECK(cudaDeviceReset());
    return exitCode;
}

int main()
{
    return runSimulation<simulationCase>();
}

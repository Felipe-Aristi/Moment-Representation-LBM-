
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

#include "../cases/channelflow/statistics.cuh"
#include "../cases/channelflow/energy_statistics.cuh"

#include "../post/channelflow/vti.cuh"

// ============================================================
// Channel-flow consistency checks
// ============================================================

static_assert(nu == channel_nu,
              "Channel flow requires nu = channel_nu in constants.cuh");

static_assert(statisticsEvery > static_cast<label_t>(0));
static_assert(output > static_cast<label_t>(0));

// ============================================================
// Main
// ============================================================

int main()
{
    constexpr int deviceID = 0;

    // ========================================================
    // Kinetic-energy monitoring
    // ========================================================

    constexpr label_t energySampleEvery =
        static_cast<label_t>(200);

    // ========================================================
    // Channel statistical output frequency
    //
    // Independent of statistics sampling and VTI output.
    // ========================================================

    constexpr label_t channelStatsWriteEvery =
        static_cast<label_t>(10000);

    CUDA_CHECK(cudaSetDevice(deviceID));

    const CudaConfig cfg =
        print_device_and_make_config(deviceID);

    // ========================================================
    // Solver memory
    // ========================================================

    Memory memory{};

    // ========================================================
    // Wall-normal statistics memory
    // ========================================================

    ChannelFlowStatistics channelStatistics{};

    // ========================================================
    // Kinetic-energy reduction memory
    // ========================================================

    StatisticsMemory energyReduction{};

    // ========================================================
    // Kinetic-energy history
    // ========================================================

    ChannelKineticEnergyHistory energyHistory{};

    // ========================================================
    // MLUPS counter
    // ========================================================

    MlupsCounter mlupsCounter{};

    bool mlupsInitialized = false;

    int exitCode = EXIT_SUCCESS;

    try
    {
        // ====================================================
        // Allocate solver memory
        // ====================================================

        allocateMemory(memory);

        // ====================================================
        // Allocate wall-normal statistics
        // ====================================================

        initializeChannelFlowStatistics(
            channelStatistics
        );

        // ====================================================
        // Allocate kinetic-energy reduction memory
        // ====================================================

        allocateStatisticsMemory(
            energyReduction
        );

        // ====================================================
        // Initialize kinetic-energy history
        //
        // Saved alongside the channel moments.
        // ====================================================

        initializeChannelKineticEnergyHistory(
            energyHistory,
            channelStatistics.directory
        );

        // ====================================================
        // Initialize MLUPS counter
        // ====================================================

        initializeMlupsCounter(mlupsCounter);
        mlupsInitialized = true;

        // ====================================================
        // Channel-flow initialization
        // ====================================================

        launchChannelFlowInitialization(
            cfg,
            memory.d_A
        );

        // ====================================================
        // A-B flip-flop pointers
        // ====================================================

        MomentState *A = &memory.d_A;
        MomentState *B = &memory.d_B;

        // ====================================================
        // Initial kinetic energy
        //
        // Record at step zero.
        // ====================================================

        const statistics_t initialEnergy =
            launchKineticEnergyStatistics(
                *A,
                energyReduction
            );

        recordChannelKineticEnergy(
            energyHistory,
            static_cast<label_t>(0),
            initialEnergy
        );

        // ====================================================
        // Initial VTI output
        // ====================================================

        copyDeviceToHost(
            memory.host,
            *A
        );

        writeChannelFlowVTI(
            memory.host,
            static_cast<label_t>(0)
        );

        // ====================================================
        // Simulation parameters
        // ====================================================

        std::cout << '\n';
        std::cout << "========================================\n";
        std::cout << "        TURBULENT CHANNEL FLOW\n";
        std::cout << "========================================\n";

        std::cout << "Re_tau: "
                  << channel_ReTau << '\n';

        std::cout << "Half-height H: "
                  << channel_halfHeight << '\n';

        std::cout << "Channel height: "
                  << channel_height << '\n';

        std::cout << "Friction velocity: "
                  << channel_uTau << '\n';

        std::cout << "Kinematic viscosity: "
                  << channel_nu << '\n';

        std::cout << "Acceleration: "
                  << channel_g << '\n';

        std::cout << "Relaxation time: "
                  << tau << '\n';

        std::cout << "Grid: "
                  << NXPhysical << " x "
                  << NY << " x "
                  << NZPhysical << '\n';

        std::cout << "Total steps: "
                  << Steps << '\n';

        std::cout << "Statistics start step: "
                  << statisticsStart << '\n';

        std::cout << "Statistics sample every: "
                  << statisticsEvery << '\n';

        std::cout << "Statistics write every: "
                  << channelStatsWriteEvery << '\n';

        std::cout << "Energy sample every: "
                  << energySampleEvery << '\n';

        std::cout << "VTI output every: "
                  << output << '\n';

        std::cout << "========================================\n\n";

        printMlupsHeader();

        // ====================================================
        // Time evolution
        // ====================================================

        for (label_t step = static_cast<label_t>(0);
             step < Steps;
             ++step)
        {
            // =================================================
            // MLUPS timer
            // =================================================

            if (step % mlupsEvery ==
                static_cast<label_t>(0))
            {
                startMlupsCounter(mlupsCounter);
            }

            // =================================================
            // Forced MR-LBM evolution
            //
            // F_x = rho * channel_g
            // F_y = 0
            // F_z = 0
            // =================================================

            launchLBMF(cfg, *A, *B);

            // =================================================
            // Channel boundary conditions
            //
            // No-slip walls at y = 0 and y = NY - 1
            // Periodicity along x and z
            // =================================================

            launchChannelBoundaryConditions(
                cfg,
                *A,
                *B
            );

            // =================================================
            // A-B swap
            // =================================================

            std::swap(A, B);

            const label_t completedStep =
                step + static_cast<label_t>(1);

            // =================================================
            // Wall-normal velocity statistics
            //
            // Fixed starting step and sampling frequency.
            // No automatic stationarity criterion.
            // =================================================

            if (statisticsSampleDue<
                    statisticsStart,
                    statisticsEvery
                >(completedStep))
            {
                if (channelStatistics.sampleCount == 0)
                {
                    const statistics_t tPlus =
                        static_cast<statistics_t>(completedStep) *
                        channel_uTau * channel_uTau /
                        channel_nu;

                    std::cout
                        << "\nStarting channel statistics at step "
                        << completedStep
                        << " | t+ = " << tPlus
                        << "\n\n";
                }

                launchChannelFlowStatistics(
                    *A,
                    channelStatistics,
                    completedStep
                );
            }

            // =================================================
            // Kinetic-energy monitoring
            //
            // Independent of wall-normal statistics.
            // The reduction returns 0.5 * sum(|u|^2).
            // =================================================

            if (completedStep % energySampleEvery ==
                static_cast<label_t>(0))
            {
                const statistics_t totalEnergy =
                    launchKineticEnergyStatistics(
                        *A,
                        energyReduction
                    );

                recordChannelKineticEnergy(
                    energyHistory,
                    completedStep,
                    totalEnergy
                );
            }

            // =================================================
            // MLUPS output
            // =================================================

            if (completedStep % mlupsEvery ==
                static_cast<label_t>(0))
            {
                printMlups(
                    mlupsCounter,
                    completedStep
                );
            }

            // =================================================
            // Periodic channel statistics output
            //
            // Only the ten wall-normal profiles are copied.
            // =================================================

            if (completedStep % channelStatsWriteEvery ==
                static_cast<label_t>(0))
            {
                if (channelStatistics.sampleCount > 0)
                {
                    writeChannelFlowStatistics(
                        channelStatistics
                    );
                }
            }

            // =================================================
            // VTI output
            // =================================================

            if (completedStep % output ==
                static_cast<label_t>(0))
            {
                copyDeviceToHost(
                    memory.host,
                    *A
                );

                writeChannelFlowVTI(
                    memory.host,
                    completedStep
                );
            }
        }

        // ====================================================
        // Final kinetic-energy measurement
        //
        // Avoid a duplicate sample at the final step.
        // ====================================================

        if (Steps % energySampleEvery !=
            static_cast<label_t>(0))
        {
            const statistics_t finalEnergy =
                launchKineticEnergyStatistics(
                    *A,
                    energyReduction
                );

            recordChannelKineticEnergy(
                energyHistory,
                Steps,
                finalEnergy
            );
        }

        // ====================================================
        // Final wall-normal statistics output
        // ====================================================

        if (channelStatistics.sampleCount > 0)
        {
            writeChannelFlowStatistics(
                channelStatistics
            );
        }

        // ====================================================
        // Final VTI output
        // ====================================================

        if (Steps % output !=
            static_cast<label_t>(0))
        {
            copyDeviceToHost(
                memory.host,
                *A
            );

            writeChannelFlowVTI(
                memory.host,
                Steps
            );
        }

        // ====================================================
        // Close kinetic-energy history
        // ====================================================

        closeChannelKineticEnergyHistory(
            energyHistory
        );

        // ====================================================
        // Final information
        // ====================================================

        std::cout << "\n========================================\n";
        std::cout << "Channel-flow simulation completed.\n";

        std::cout << "Statistical samples: "
                  << channelStatistics.sampleCount << '\n';

        std::cout << "Energy samples: "
                  << energyHistory.sampleCount << '\n';

        std::cout << "First statistics step: "
                  << channelStatistics.firstStep << '\n';

        std::cout << "Last statistics step: "
                  << channelStatistics.lastStep << '\n';

        std::cout << "Statistics directory: "
                  << channelStatistics.directory << '\n';

        std::cout << "========================================\n";
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "Channel-flow error: "
            << error.what() << '\n';

        exitCode = EXIT_FAILURE;
    }

    // ========================================================
    // Cleanup
    // ========================================================

    CUDA_CHECK(cudaDeviceSynchronize());

    if (energyHistory.csv.is_open())
    {
        energyHistory.csv.close();
    }

    if (mlupsInitialized)
    {
        destroyMlupsCounter(mlupsCounter);
    }

    freeStatisticsMemory(energyReduction);

    freeChannelFlowStatistics(channelStatistics);

    freeMemory(memory);

    CUDA_CHECK(cudaDeviceReset());

    return exitCode;
}

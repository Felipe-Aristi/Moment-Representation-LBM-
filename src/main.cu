
#include <cuda_runtime.h>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <utility>

#include "constants.cuh"
#include "memory.cuh"
#include "launch.cuh"

#include "statistics/statisticsMemory.cuh"

#include "utilities/cudaConfig.cuh"
#include "utilities/cudaUtilities.cuh"
#include "utilities/types.cuh"
#include "utilities/mlups_count.cuh"

#include "../cases/jet/statistics.cuh"
#include "../cases/jet/energy_statistics.cuh"

#include "../post/jet/vti.cuh"

int main()
{
    constexpr int deviceID = 0;

    // ============================================================
    // Energy monitoring frequency
    // ============================================================

    constexpr label_t energySampleEvery = static_cast<label_t>(2000);

    CUDA_CHECK(cudaSetDevice(deviceID));

    const CudaConfig cfg = print_device_and_make_config(deviceID);

    // ============================================================
    // Solver memory
    // ============================================================

    Memory memory;
    allocateMemory(memory);

    // ============================================================
    // Radial statistics memory
    // ============================================================

    JetRadialStatistics jetStatistics;
    initializeJetRadialStatistics(jetStatistics);

    // ============================================================
    // Kinetic-energy reduction memory
    // ============================================================

    StatisticsMemory energyReduction;
    allocateStatisticsMemory(energyReduction);

    // ============================================================
    // MLUPS counter
    // ============================================================

    MlupsCounter mlupsCounter;
    initializeMlupsCounter(mlupsCounter);
    printMlupsHeader();

    int exitCode = EXIT_SUCCESS;

    try
    {
        // ========================================================
        // Kinetic-energy output
        // ========================================================

        JetKineticEnergyHistory energyHistory;
        initializeJetKineticEnergyHistory(energyHistory);

        // ========================================================
        // Turbulent jet initialization
        // ========================================================

        launchJetInitialization(cfg, memory.d_A);

        // ========================================================
        // A-B flip-flop pointers
        // ========================================================

        MomentState *A = &memory.d_A;
        MomentState *B = &memory.d_B;

        // ========================================================
        // Initial kinetic energy
        // ========================================================

        const statistics_t initialEnergy =
            launchKineticEnergyStatistics(*A, energyReduction);

        recordJetKineticEnergy(
            energyHistory,
            static_cast<label_t>(0),
            initialEnergy
        );

        // ========================================================
        // Initialize radial output files
        // ========================================================

        writeJetRadialStatistics(jetStatistics);

        // ========================================================
        // Initial VTI output
        // ========================================================

        copyDeviceToHost(memory.host, *A);
        writeJetVTI(memory.host, static_cast<label_t>(0));

        std::cout << "Jet Reynolds number: " << jet_Re << '\n';
        std::cout << "Jet diameter: " << jet_diameter << '\n';
        std::cout << "Jet velocity: " << jet_velocity << '\n';
        std::cout << "Statistics start t*: " << jetStatsStartTStar << '\n';
        std::cout << "Statistics sampling interval: "
                  << jetStatsSampleEvery << '\n';
        std::cout << "Energy sampling interval: "
                  << energySampleEvery << '\n';

        // ========================================================
        // Time evolution
        // ========================================================

        for (label_t step = static_cast<label_t>(0); step < Steps; ++step)
        {
            // ====================================================
            // MLUPS timer
            // ====================================================

            if (step % mlupsEvery == static_cast<label_t>(0))
            {
                startMlupsCounter(mlupsCounter);
            }

            // ====================================================
            // MR-LBM evolution
            // ====================================================

            launchLBM(cfg, *A, *B);

            // ====================================================
            // Jet boundary conditions
            // ====================================================

            launchJetBoundaryConditions(cfg, *A, *B);

            // ====================================================
            // A-B swap
            // ====================================================

            std::swap(A, B);

            const label_t completedStep =
                step + static_cast<label_t>(1);

            // ====================================================
            // Dimensionless time
            // ====================================================

            const real_t tStar =
                static_cast<real_t>(completedStep) *
                jet_velocity / jet_diameter;

            // ====================================================
            // Kinetic-energy monitoring
            // ====================================================

            if (completedStep % energySampleEvery ==
                static_cast<label_t>(0))
            {
                const statistics_t totalEnergy =
                    launchKineticEnergyStatistics(
                        *A, energyReduction
                    );

                recordJetKineticEnergy(
                    energyHistory,
                    completedStep,
                    totalEnergy
                );
            }

            // ====================================================
            // Radial statistics
            //
            // Begin at the prescribed dimensionless time.
            // Sample at a fixed frequency.
            // ====================================================

            if (tStar >= jetStatsStartTStar &&
                completedStep % jetStatsSampleEvery ==
                    static_cast<label_t>(0))
            {
                if (jetStatistics.sampledSteps == 0)
                {
                    std::cout
                        << "Starting jet radial statistics at step "
                        << completedStep
                        << " | t* = " << tStar << '\n';
                }

                launchJetRadialStatistics(
                    cfg,
                    *A,
                    jetStatistics,
                    completedStep
                );
            }

            // ====================================================
            // MLUPS output
            // ====================================================

            if (completedStep % mlupsEvery ==
                static_cast<label_t>(0))
            {
                printMlups(mlupsCounter, completedStep);
            }

            // ====================================================
            // Statistical output and visualization
            // ====================================================

            if (completedStep % output ==
                static_cast<label_t>(0))
            {
                // Save radial moments after sampling begins
                if (jetStatistics.sampledSteps > 0)
                {
                    writeJetRadialStatistics(jetStatistics);
                }

                // Write jet velocity field
                copyDeviceToHost(memory.host, *A);
                writeJetVTI(memory.host, completedStep);
            }
        }

        // ========================================================
        // Final kinetic-energy measurement
        // ========================================================

        if (Steps % energySampleEvery != static_cast<label_t>(0))
        {
            const statistics_t finalEnergy =
                launchKineticEnergyStatistics(*A, energyReduction);

            recordJetKineticEnergy(
                energyHistory,
                Steps,
                finalEnergy
            );
        }

        // ========================================================
        // Final radial statistics output
        // ========================================================

        writeJetRadialStatistics(jetStatistics);

        // ========================================================
        // Close kinetic-energy output
        // ========================================================

        closeJetKineticEnergyHistory(energyHistory);

        // ========================================================
        // Final information
        // ========================================================

        std::cout << "Jet simulation completed.\n";
        std::cout << "Radial samples: "
                  << jetStatistics.sampledSteps << '\n';
        std::cout << "Radial statistics directory: "
                  << jetStatistics.directory << '\n';
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        exitCode = EXIT_FAILURE;
    }

    // ============================================================
    // Cleanup
    // ============================================================

    CUDA_CHECK(cudaDeviceSynchronize());

    destroyMlupsCounter(mlupsCounter);

    freeStatisticsMemory(energyReduction);
    freeJetRadialStatistics(jetStatistics);
    freeMemory(memory);

    CUDA_CHECK(cudaDeviceReset());

    return exitCode;
}

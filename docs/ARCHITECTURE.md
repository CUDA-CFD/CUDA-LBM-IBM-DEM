# Architecture

## CPU production path

- `LBMSolver`: D2Q9 BGK fluid solver with Guo forcing.
- `IBMParticle`: rigid circular particle state and Lagrangian markers.
- `IBMCoupling`: interpolation, direct forcing, spreading, force and torque
  accumulation.
- `Simulation`: sedimentation timestep orchestration.
- `WallContact`: four-wall linear spring-dashpot contact.
- `VTKWriter`: ParaView output.

## CUDA production path under construction

- `LBMSolverCUDA`: persistent device-resident fluid state.
- `IBMParticleCUDA`: persistent device-resident particle and marker state.
- Independent CUDA launchers retain one-to-one mappings to the CPU LBM and IBM
  operations.
- `--verify-ibm-pipeline` compares the connected hydrodynamic GPU path with the
  CPU implementation over repeated timesteps.

Gravity, buoyancy, wall contact, benchmark output, and the final CUDA
sedimentation runner are intentionally deferred to subsequent commits.


# CUDA-LBM-IBM-DEM Software Architecture

## Overview

CUDA-LBM-IBM-DEM is organized into four major layers:

1. User Interface
2. Simulation Engine
3. Verification Framework
4. Validation and Benchmark Tools

The CPU and CUDA implementations share the same numerical algorithms.
The CUDA implementation replaces the computational kernels while preserving
the same mathematical formulation and verification workflow.


```mermaid
flowchart TD

    User([User])

    User --> CLI[Command Line]

    CLI --> Main[main.cpp]

    Main --> CPU[CPU Solver]

    Main --> GPU[CUDA Solver]

    Main --> Verify[Verification]

    Main --> Bench[Benchmark]

    CPU --> Output

    GPU --> Output

    Verify --> Output

    Bench --> Output

    Output --> CSV[CSV]

    Output --> VTK[VTK]

    Output --> Plots[Validation & Plots]
```

# Figure 1 — Overall software architecture

```mermaid
flowchart TD

Start([Start])

Start --> Init

Init[Initialisation]

Init --> Macro

Macro[Macroscopic Variables]

Macro --> Interp

Interp[IBM Interpolation]

Interp --> Force

Force[Direct Forcing]

Force --> Spread

Spread[Force Spreading]

Spread --> Hydro

Hydro[Hydrodynamic Loads]

Hydro --> External

External[Gravity + Wall Contact]

External --> Collision

Collision --> Streaming

Streaming --> Integrate

Integrate[Particle Integration]

Integrate --> Marker

Marker[Marker Update]

Marker --> Output

Output --> End([Next Timestep])
```
# Figure 2 — CPU execution pipeline

```mermaid
flowchart TD

Host[Host]

Host --> CopyH2D

CopyH2D[Initial Host → Device Copy]

CopyH2D --> Macro

Macro[launchComputeMacroscopic]

Macro --> Interp

Interp[launchIBMInterpolation]

Interp --> Force

Force[launchIBMForcing]

Force --> Spread

Spread[launchIBMSpreading]

Spread --> Hydro

Hydro[launchHydrodynamicLoads]

Hydro --> Loads

Loads[launchExternalLoads]

Loads --> Collision

Collision[launchCollideAndForce]

Collision --> Streaming

Streaming[launchPullStreaming]

Streaming --> Integrate

Integrate[launchParticleIntegration]

Integrate --> Output

Output{Output Interval?}

Output -->|No| Macro

Output -->|Yes| CopyBack

CopyBack[Device → Host]

CopyBack --> WriteFiles

WriteFiles --> Macro
```
# Figure 3 — CUDA execution pipeline

```mermaid
flowchart LR

    Host[Host Memory]

    subgraph GPU["Persistent GPU Memory"]
        DeviceState["Device-resident simulation state"]

        Pop[D2Q9 Populations]
        Density[Density]
        Velocity[Velocity]
        EulerianForce[Eulerian Forces]
        MarkerPos[Marker Positions]
        MarkerVel[Marker Velocities]
        MarkerForce[Marker Forces]
        Particle[Particle State]

        DeviceState --> Pop
        DeviceState --> Density
        DeviceState --> Velocity
        DeviceState --> EulerianForce
        DeviceState --> MarkerPos
        DeviceState --> MarkerVel
        DeviceState --> MarkerForce
        DeviceState --> Particle
    end

    Host -->|Initial host-to-device copy| DeviceState
    DeviceState -->|Selected output and
verification copies| Host
```

# Figure 4 — Persistent GPU memory

```mermaid
flowchart TD

Main[main.cpp]

Main --> CPU

Main --> CUDA

CPU --> Simulation

Simulation --> LBM

Simulation --> IBM

Simulation --> Particle

CUDA --> CUDAUtils

CUDAUtils --> Macro

Macro --> Collision

Collision --> Streaming

Streaming --> Interpolation

Interpolation --> Forcing

Forcing --> Spreading

Spreading --> Hydro

Hydro --> External

External --> Integration
```
# Figure 5 — Source-code dependencies


```mermaid
flowchart TD

CPU[CPU Reference]

CUDA[CUDA Kernel]

CPU --> Compare

CUDA --> Compare

Compare --> Error

Error[Max Error<br/>Relative L2]

Error --> Decision{Tolerance?}

Decision -->|Pass| PASS

Decision -->|Fail| FAIL
```
# Figure 6 — Verification workflow

```mermaid
flowchart TD

Interpolation

Forcing

Spreading

Hydro

Integration

Pipeline

Sedimentation

Interpolation --> Pipeline

Forcing --> Pipeline

Spreading --> Pipeline

Hydro --> Pipeline

Integration --> Pipeline

Pipeline --> Sedimentation
```
# Figure 7 — Complete verification hierarchy

```mermaid
flowchart TD

Repo[CUDA-LBM-IBM-DEM]

Repo --> src

Repo --> cuda

Repo --> include

Repo --> tests

Repo --> docs

Repo --> benchmark

Repo --> tools

Repo --> results

Repo --> CMake

Repo --> README
```

```mermaid
flowchart TD

Repo[CUDA-LBM-IBM-DEM]

Repo --> src

Repo --> cuda

Repo --> include

Repo --> tests

Repo --> docs

Repo --> benchmark

Repo --> tools

Repo --> results

Repo --> CMake

Repo --> README
```

# Figure 8 — Repository organization

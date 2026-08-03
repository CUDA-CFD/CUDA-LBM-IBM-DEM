# CUDA streaming kernel

## Purpose

The streaming operation transports each post-collision D2Q9 population to the neighboring lattice node associated with its discrete velocity.

## Thread ownership

One CUDA thread owns one destination lattice node.

For each of the nine directions, the thread identifies the neighboring source node and gathers the required population. This is a **pull (gather) streaming** scheme.

## Data access

Each thread:

- reads up to nine populations from neighboring nodes;
- reads the local density only when a moving-wall correction is required;
- writes all nine populations belonging to its own destination node.

Because each thread writes only to its own node, the kernel has no write races and requires no atomics.

## Boundary conditions

The implementation reproduces the CPU boundary behavior:

- periodic top/bottom boundaries for the sedimentation benchmark;
- halfway bounce-back at the left and right walls;
- stationary bounce-back on all walls for a closed cavity;
- optional moving top-wall correction for the lid-driven cavity mode.

## Verification

The verification routine constructs the pre-stream state using the production CPU collision function. It then streams that identical state with:

1. the CPU push-streaming implementation; and
2. the CUDA pull-streaming implementation.

All nine populations at every node are compared using the maximum absolute error.

#pragma once

namespace cuda_backend
{

/**
 * @brief Run the complete gravity-driven sedimentation benchmark on the GPU.
 *
 * The LBM fields, IBM marker arrays, and rigid-particle state remain on the
 * device between timesteps. A compact particle/load snapshot is copied to the
 * host for each CSV row, while full velocity and marker arrays are downloaded
 * only at the requested VTK output interval.
 */
int runCudaSedimentation(int steps, int outputInterval);

} // namespace cuda_backend

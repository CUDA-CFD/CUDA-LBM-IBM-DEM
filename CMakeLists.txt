cmake_minimum_required(VERSION 3.20)

project(
    CudaLBM_IBM_DEM
    VERSION 0.1.0
    LANGUAGES CXX
)

option(ENABLE_CUDA "Build the optional CUDA backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CPU_SOURCES
    src/main.cpp
    src/LBMSolver.cpp
    src/LBMInitialize.cpp
    src/LBMMacroscopic.cpp
    src/LBMCollision.cpp
    src/LBMStreaming.cpp
    src/IBMParticle.cpp
    src/IBMCoupling.cpp
    src/IBMInterpolation.cpp
    src/IBMForcing.cpp
    src/IBMSpreading.cpp
    src/VTKWriter.cpp
    src/WallContact.cpp
    src/Simulation.cpp
)

add_executable(lbm_ibm_dem ${CPU_SOURCES})

target_include_directories(
    lbm_ibm_dem
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "ENABLE_CUDA=ON, but no CUDA compiler was detected."
        )
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # NVIDIA Ada Lovelace / RTX 4090. Override from the command line
        # for other GPUs, e.g. -DCMAKE_CUDA_ARCHITECTURES=86.
        set(CMAKE_CUDA_ARCHITECTURES 89)
    endif()

    target_sources(
        lbm_ibm_dem
        PRIVATE
        cuda/CUDAUtils.cu
        cuda/LBMMacroscopic.cu
        cuda/LBMCollision.cu
        cuda/LBMStreaming.cu
        cuda/LBMSolverCUDA.cu
        cuda/GPULBMRunner.cu
        cuda/GPUVerification.cu
        cuda/IBMInterpolation.cu
        cuda/IBMInterpolationVerification.cu
        cuda/IBMForcing.cu
        cuda/IBMForcingVerification.cu
        cuda/IBMSpreading.cu
        cuda/IBMSpreadingVerification.cu
        tests/VerificationMetrics.cpp
    )

    target_include_directories(
        lbm_ibm_dem
        PRIVATE
        ${PROJECT_SOURCE_DIR}/cuda
        ${PROJECT_SOURCE_DIR}/tests
    )

    target_compile_definitions(
        lbm_ibm_dem
        PRIVATE
        USE_CUDA=1
    )

    target_link_libraries(
        lbm_ibm_dem
        PRIVATE
        CUDA::cudart
    )

    set_target_properties(
        lbm_ibm_dem
        PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )

    message(STATUS "CUDA backend: ENABLED")
    message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA backend: DISABLED (CPU-only build)")
endif()

if(MSVC)
    target_compile_options(lbm_ibm_dem PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(
        lbm_ibm_dem
        PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_custom_target(validate_sedimentation
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tools/validate_sedimentation.py
                --simulation ${PROJECT_SOURCE_DIR}/sedimentation.csv
                --reference ${PROJECT_SOURCE_DIR}/benchmark/sedimentation/reference_position_N100.csv
                --output-dir ${PROJECT_SOURCE_DIR}/results/sedimentation_validation
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Comparing sedimentation.csv with the published N=100 reference data"
        VERBATIM
    )
endif()

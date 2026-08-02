cmake_minimum_required(VERSION 3.20)
project(CudaLBM_IBM_DEM VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(lbm_ibm_dem
    src/main.cpp
    src/LBMSolver.cpp
    src/IBMParticle.cpp
    src/IBMCoupling.cpp
    src/VTKWriter.cpp
    src/Simulation.cpp
)

target_include_directories(lbm_ibm_dem PRIVATE ${PROJECT_SOURCE_DIR}/include)

if(MSVC)
  target_compile_options(lbm_ibm_dem PRIVATE /W4)
else()
  target_compile_options(lbm_ibm_dem PRIVATE -Wall -Wextra -Wpedantic)
endif()

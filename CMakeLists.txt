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
    src/WallContact.cpp
    src/Simulation.cpp
)

target_include_directories(lbm_ibm_dem PRIVATE ${PROJECT_SOURCE_DIR}/include)

if(MSVC)
  target_compile_options(lbm_ibm_dem PRIVATE /W4)
else()
  target_compile_options(lbm_ibm_dem PRIVATE -Wall -Wextra -Wpedantic)
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

# ggemm: grouped FP64 GEMM emulated on INT8 tensor cores (owner: agent-P)
# Build (login node: compile only):
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 12
cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90 120)   # H100/H200 (dev) + RTX 5090 (target, nano4)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project(ggemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDAToolkit_ROOT /usr/local/cuda-13.0)
find_package(CUDAToolkit REQUIRED)

# ---- module sources ------------------------------------------------------------------
set(GGEMM_SRC src/ggemm.cu src/planner.cpp)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/ozaki2_numerics.cu)
  list(APPEND GGEMM_SRC src/ozaki2_numerics.cu)
else()
  message(FATAL_ERROR "ggemm: src/ozaki2_numerics.cu missing (agent-N)")
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/int8_grouped_gemm.cu)
  list(APPEND GGEMM_SRC src/int8_grouped_gemm.cu)
else()
  message(FATAL_ERROR "ggemm: src/int8_grouped_gemm.cu missing (agent-K)")
endif()

add_library(ggemm STATIC ${GGEMM_SRC})
target_include_directories(ggemm
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ggemm PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt)
# planner auto-loads bench/calib/params_<gpu_slug>.json (override: env GGEMM_CALIB_JSON)
target_compile_definitions(ggemm PRIVATE GGEMM_CALIB_DIR="${CMAKE_CURRENT_SOURCE_DIR}/bench/calib")
set_target_properties(ggemm PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
                                       POSITION_INDEPENDENT_CODE ON)
target_compile_options(ggemm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr -Xcompiler=-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>)

# ---- tests / bench (owner of the files: agent-B) ---------------------------------------
# Convention: tests/test_*.cu and bench/bench_*.cu each hold a main() -> one executable each.
# Every other .cu/.cpp under tests/ and bench/ is shared support code (ggemm_support).
file(GLOB GGEMM_SUPPORT_SRC CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cpp)
list(FILTER GGEMM_SUPPORT_SRC EXCLUDE REGEX "/(test|bench)_[^/]*$")
file(GLOB GGEMM_TEST_MAINS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cu)
file(GLOB GGEMM_BENCH_MAINS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bench/bench_*.cu)

set(GGEMM_SUPPORT_LIB "")
if(GGEMM_SUPPORT_SRC)
  add_library(ggemm_support STATIC ${GGEMM_SUPPORT_SRC})
  target_include_directories(ggemm_support PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/bench)
  target_link_libraries(ggemm_support PUBLIC ggemm)
  set(GGEMM_SUPPORT_LIB ggemm_support)
endif()

# Tests/bench may poke at internals (planner, module APIs): give them src/ includes.
set(GGEMM_INTERNAL_INC ${CMAKE_CURRENT_SOURCE_DIR}/src)

enable_testing()
foreach(f ${GGEMM_TEST_MAINS})
  get_filename_component(name ${f} NAME_WE)
  add_executable(${name} ${f})
  target_include_directories(${name} PRIVATE ${GGEMM_INTERNAL_INC})
  target_link_libraries(${name} PRIVATE ggemm ${GGEMM_SUPPORT_LIB})
  add_test(NAME ${name} COMMAND ${name})
endforeach()
foreach(f ${GGEMM_BENCH_MAINS})
  get_filename_component(name ${f} NAME_WE)
  add_executable(${name} ${f})
  target_include_directories(${name} PRIVATE ${GGEMM_INTERNAL_INC})
  target_link_libraries(${name} PRIVATE ggemm ${GGEMM_SUPPORT_LIB})
endforeach()

add_custom_target(tests)
add_custom_target(bench)
if(GGEMM_SUPPORT_LIB)
  add_dependencies(tests ${GGEMM_SUPPORT_LIB})
  add_dependencies(bench ${GGEMM_SUPPORT_LIB})
endif()
foreach(f ${GGEMM_TEST_MAINS})
  get_filename_component(name ${f} NAME_WE)
  add_dependencies(tests ${name})
endforeach()
foreach(f ${GGEMM_BENCH_MAINS})
  get_filename_component(name ${f} NAME_WE)
  add_dependencies(bench ${name})
endforeach()

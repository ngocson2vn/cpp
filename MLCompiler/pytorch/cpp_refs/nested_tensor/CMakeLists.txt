cmake_minimum_required(VERSION 3.10)
project(nested_tensor)
set(CMAKE_CXX_STANDARD 17)

message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
include_directories(${PROJECT_SOURCE_DIR})

message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

message(STATUS "CUDA_ROOT: ${CUDA_ROOT}")
include_directories(${CUDA_ROOT}/include)

# Find the CUDA runtime library
find_library(CUDA_RUNTIME_LIBRARY
  NAMES cudart
  PATHS ${CUDA_ROOT}/lib64
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
)
message(STATUS "CUDA_RUNTIME_LIBRARY: ${CUDA_RUNTIME_LIBRARY}")

# Find the CUDA driver library
find_library(CUDA_DRIVER_LIBRARY
  NAMES cuda
  PATHS /usr/lib/x86_64-linux-gnu
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
)
message(STATUS "CUDA_DRIVER_LIBRARY: ${CUDA_DRIVER_LIBRARY}")

# Torch
find_package(Torch REQUIRED)
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")


add_executable(main 
  main.cpp
)
target_compile_options(main PRIVATE -g -O0)
target_link_options(main PUBLIC -rdynamic)
target_link_libraries(main PUBLIC
  ${CUDA_DRIVER_LIBRARY}
  ${TORCH_LIBRARIES}
)

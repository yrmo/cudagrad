cmake_minimum_required(VERSION 3.16)
project(pybind11_cuda_project)

find_package(CUDA)
if (CUDA_FOUND)
    enable_language(CUDA)
    set(USE_CUDA ON)
    set(CMAKE_CUDA_ARCHITECTURES 60)
else()
    set(USE_CUDA OFF)
endif()

if (USE_CUDA)
    message(STATUS "Building with CUDA support")
else()
    message(STATUS "Building without CUDA support (CPU-only)")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})
add_subdirectory(pybind11)
include_directories(./pybind11/include)

if (USE_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cccl/thrust)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cccl/libcudacxx/include)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cccl/cub)
    link_directories(${CUDA_LIBRARY_DIRS})
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

set(SOURCE_FILES src/bindings.cpp src/operations.cu src/operations.cpp src/tensor.hpp)

add_library(tensor MODULE ${SOURCE_FILES})

if (USE_CUDA)
    set_target_properties(tensor PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(tensor ${Python3_LIBRARIES} ${CUDA_LIBRARIES} pybind11::module)
else()
    target_link_libraries(tensor ${Python3_LIBRARIES} pybind11::module)
endif()

if (USE_CUDA)
    target_compile_definitions(tensor PRIVATE CUDA_ENABLED=1)
else()
    target_compile_definitions(tensor PRIVATE CUDA_ENABLED=0)
endif()

set_target_properties(tensor PROPERTIES PREFIX "")

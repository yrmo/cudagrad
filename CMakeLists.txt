cmake_minimum_required(VERSION 3.18)
project(pybind11_cuda_project)

enable_language(CUDA)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cccl/thrust)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cccl/libcudacxx/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cccl/cub)

link_directories(${CUDA_LIBRARY_DIRS})

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})
add_subdirectory(pybind11)
include_directories(./pybind11/include)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "86")

set(SOURCE_FILES src/bindings.cpp src/operations.cu src/tensor.hpp)
add_library(tensor SHARED ${SOURCE_FILES})
set_target_properties(tensor PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(tensor ${Python3_LIBRARIES} ${CUDA_LIBRARIES} pybind11::module)
set_target_properties(tensor PROPERTIES PREFIX "")

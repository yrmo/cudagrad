cmake_minimum_required(VERSION 3.12)
project(pybind11_cuda_project)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})

add_subdirectory(pybind11)
include_directories(./pybind11/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# Add your source files from the src directory
set(SOURCE_FILES src/bindings.cpp src/operations.cu src/tensor.hpp) # Add any other files as needed

CUDA_ADD_LIBRARY(tensor SHARED ${SOURCE_FILES})
target_link_libraries(tensor ${Python3_LIBRARIES} ${CUDA_LIBRARIES} pybind11::module)
set_target_properties(tensor PROPERTIES PREFIX "")

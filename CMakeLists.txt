cmake_minimum_required(VERSION 3.18)
project(pybind11_cuda_project)

set(SOURCE_FILES src/bindings.cpp src/tensor.hpp)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})
add_subdirectory(pybind11)
include_directories(./pybind11/include)

set(CMAKE_CXX_STANDARD 17)

add_library(tensor MODULE ${SOURCE_FILES})

target_link_libraries(tensor ${Python3_LIBRARIES} pybind11::module)
set_target_properties(tensor PROPERTIES PREFIX "")

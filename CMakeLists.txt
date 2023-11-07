cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(tensor_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build googletest from the submodule
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/googletest)

# Add the target executable
add_executable(${PROJECT_NAME} tests/test.cpp)

# Link the executable against the googletest
target_link_libraries(${PROJECT_NAME} gtest gtest_main)

# Add compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)

# Copy headers to build directory
configure_file(src/tensor.hpp tensor.hpp COPYONLY)

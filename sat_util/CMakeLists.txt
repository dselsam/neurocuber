cmake_minimum_required(VERSION 2.8.12)
project(pysat)

set(CMAKE_CXX_FLAGS                "-Wall -Wextra -std=c++14")
set(CMAKE_CXX_FLAGS_DEBUG          "-g3")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG")

set(PYBIND11_CPP_STANDARD -std=c++14)
set(PYBIND11_CPP_STANDARD -O2)

find_package(pybind11 REQUIRED)
include_directories(/usr/local/include/eigen3)

pybind11_add_module(sat_util problem.cpp solver.cpp sat_util.cpp)
target_link_libraries(sat_util PUBLIC libz3.so)

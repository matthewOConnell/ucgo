cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(UCGO)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(Kokkos)
add_subdirectory(vul)

add_subdirectory(test)


cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(UCGO)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(Kokkos_ENABLE_SERIAL "Enable Kokkos Serial backend" ON)
if (Kokkos_ENABLE_CUDA)
  option(Kokkos_ENABLE_CUDA_LAMBDA "Enable CUDA Lambda support" ON)
endif()

if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/Kokkos)
  message(FATAL_ERROR "Kokkos missing from top-level.\nDownload Kokkos from Github, via `git clone https://github.com/kokkos/kokkos.git Kokkos`")
endif()

find_package(ddata QUIET)
add_subdirectory(Kokkos)
add_subdirectory(vul)
add_subdirectory(test)

#!/usr/bin/env bash
module load  cmake_3.18.0 cuda_11.2.2 gcc_9.2.0  
rm -rf build-nvcc
mkdir build-nvcc
cd build-nvcc
cmake -DCMAKE_CXX_COMPILER=/hpnobackup1/mdoconn1/ucgo-mini-app/nvcc_wrapper \
      -DCMAKE_CXX_FLAGS="-lineinfo" \
      -DKokkos_ARCH_VOLTA70=ON \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
      -DKokkos_ENABLE_CUDA_LAMBDA=ON \
      ..
make -j

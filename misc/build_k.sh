#!/usr/bin/env bash

set -u
set -e

build_cpu() {
  local build_dir=$1
  mkdir -p $build_dir
  cd $build_dir
    cmake  \
             -DCMAKE_INSTALL_PREFIX=$PWD \
             -DKokkos_ENABLE_OPENMP=ON \
             -DCMAKE_CXX_COMPILER=g++ \
             ..
    
    make -j4 install
  cd ..
}

build_gpu() {
  local build_dir=$1
  local nvcc_wrapper=${PWD}/Kokkos/bin/nvcc_wrapper
  mkdir -p $build_dir
  cd $build_dir
    cmake  \
             -DCMAKE_INSTALL_PREFIX=$PWD \
             -DCMAKE_CXX_COMPILER=${nvcc_wrapper} \
             -DKokkos_ENABLE_CUDA=ON \
             -DKokkos_ENABLE_OPENMP=ON \
             -DKokkos_ARCH_VOLTA70=ON \
             ..
    
    make -j4 install
  cd ..
}

module purge
module load cuda_11.2.2 gcc_9.2.0 cmake_3.18.0


if [[ ! -d Kokkos ]]; then
  git clone https://github.com/kokkos/kokkos.git Kokkos
fi

# build for host
build_cpu "cpu_build"

# build for NVIDIA V100 GPU
build_gpu "gpu_build"

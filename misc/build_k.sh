#!/usr/bin/env bash

set -u
set -e

build_cpu() {
  local build_dir=$1
  mkdir -p $build_dir
  cd $build_dir
    cmake  \
             -DCMAKE_INSTALL_PREFIX=$PWD \
             -DCMAKE_CXX_COMPILER=g++ \
             ..
    
    make -j$(nproc) install
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
    
    make -j$(nproc) install
  cd ..
}

if [[ ! -d Kokkos ]]; then
  git clone https://github.com/kokkos/kokkos.git Kokkos
fi

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source $SCRIPT_DIR/k-env

# build for host
build_cpu "cpu_build"

# build for NVIDIA V100 GPU
build_gpu "gpu_build"

#!/usr/bin/env bash
module load  cuda_11.2.2 gcc_9.2.0   clang_12.0.0   
module list
export KOKKOS_PROFILE_LIBRARY=/lustre2/hpnobackup1/mdoconn1/kokkos-tools/kp_nvprof_connector.so
nsys profile --force-overwrite=true -o out.profile -t cuda,nvtx ./build-nvcc/vul/ucgo -g 10 10 100
#./build-gpu/vul/ucgo

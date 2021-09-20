#!/usr/bin/env bash
module use /swbuild/kbthomp1/modulefiles
module purge
module load cuda/11.2.2 kokkos-tools/1.0
module list

export KOKKOS_PROFILE_LIBRARY=${KOKKOS_TOOLS_ROOT}/kp_nvprof_connector.so
nsys profile --force-overwrite=true -t cuda,nvtx ./gpu_build/bin/ucgo -g 10 10 100

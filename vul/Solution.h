#pragma once
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

namespace vul {
template <int N> using SolutionArray = Kokkos::DualView<double *[N]>;
template <int N> using StaticArray = Kokkos::Array<double, N>;
}
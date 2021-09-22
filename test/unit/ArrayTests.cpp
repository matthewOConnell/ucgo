#include <vul/Solution.h>
#include <catch.hpp>

TEST_CASE("Compute norm of array"){
  Kokkos::View<double*> s("s", 100);
  for(int c = 0; c < 100; c++){
    s(c) = 1.0;
  }

  double norm;
  auto calc_norm = KOKKOS_LAMBDA(int c, double& norm){
    norm += s(c)*s(c);
  };

  Kokkos::parallel_reduce(100, calc_norm, norm);
  norm = sqrt(norm);

  REQUIRE(norm == sqrt(100));

}

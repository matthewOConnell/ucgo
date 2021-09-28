#include <catch.hpp>
#include <ddata/ETD.h>

using AD = Linearize::ETD<5>;

TEST_CASE("Can compute the flux Jacobian") {
  std::array<double, 5> v = {1, 2, 3, 4, 5};
  auto v_ddt              = AD::Identity(v);
  auto jacobian           = Linearize::ExtractBlockJacobian(v_ddt);
  for (int r = 0; r < 5; ++r) {
    for (int c = 0; c < 5; ++c) {
      double expected = r == c ? 1.0 : 0.0;
      REQUIRE(expected == jacobian[r * 5 + c]);
    }
  }
}

#include <catch.hpp>
#include <vul/Solution.h>
#include <vul/PerfectGas.h>


TEST_CASE("Can compute gas variables from state") {
  double u        = 343.21;
  double v        = 0.0;
  double w        = 0.0;
  double pressure = 101325;
  double density  = 1.225;
  double gamma    = 1.4;
  double total_energy =
      vul::perfect_gas::calcTotalEnergy(density, u, v, w, pressure, gamma);

  vul::StaticArray<5> q{density, density*u, density*v, density*w, total_energy};
  double calc_press = vul::perfect_gas::calcPressure(q, gamma);
  REQUIRE(pressure == Approx(calc_press));
}

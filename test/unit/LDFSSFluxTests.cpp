#include <doctest.h>
#include <vul/Flux.h>
#include <vul/PhysicalFlux.h>
#include <vul/PerfectGas.h>

TEST_CASE("can compute LDFSS flux") {
  vul::Point<double> n = {1.0/sqrt(3.0), 1.0/sqrt(3.0), 1.0/sqrt(3)};
  double gamma = 1.4;

  SUBCASE("constant state") {
    auto q = vul::StaticArray<5>{1.1, 0.1, 0.2, 0.3, 2.5};
    auto qg = vul::StaticArray<2>{gamma, vul::perfect_gas::calcPressure(q, gamma)};
    auto consistent_flux = vul::PhysicalFlux::inviscidFlux<5, 2>(q, qg, n);
    auto flux = vul::LDFSSFlux::inviscidFlux(q, q, qg, qg, n);
    REQUIRE(consistent_flux[0] == flux[0]);
    REQUIRE(consistent_flux[1] == flux[1]);
    REQUIRE(consistent_flux[2] == flux[2]);
    REQUIRE(consistent_flux[3] == flux[3]);
    REQUIRE(consistent_flux[4] == flux[4]);
  }
  SUBCASE("supersonic - upwind left state") {
    auto ql = vul::StaticArray<5>{1.1, 12.1, 12.2, 12.3, 3.5};
    auto qr = vul::StaticArray<5>{1.0, 10.1, 10.2, 10.3, 2.5};
    auto qgl = vul::StaticArray<2>{gamma, vul::perfect_gas::calcPressure(ql, gamma)};
    auto qgr = vul::StaticArray<2>{gamma, vul::perfect_gas::calcPressure(qr, gamma)};
    auto consistent_flux =
        vul::PhysicalFlux::inviscidFlux(ql, qgl, n);
    auto flux = vul::LDFSSFlux::inviscidFlux(ql, qr, qgl, qgr, n);
    REQUIRE(consistent_flux[0] == flux[0]);
    REQUIRE(consistent_flux[1] == flux[1]);
    REQUIRE(consistent_flux[2] == flux[2]);
    REQUIRE(consistent_flux[3] == flux[3]);
    REQUIRE(consistent_flux[4] == flux[4]);
  }
  SUBCASE("supersonic - upwind right state") {
    auto ql = vul::StaticArray<5>({1.0, -10.1, -10.2, -10.3, 2.0});
    auto qr = vul::StaticArray<5>({1.1, -12.1, -12.2, -12.3, 3.0});
    auto qgl = vul::StaticArray<2>{gamma, vul::perfect_gas::calcPressure(ql, gamma)};
    auto qgr = vul::StaticArray<2>{gamma, vul::perfect_gas::calcPressure(qr, gamma)};
    auto consistent_flux =
        vul::PhysicalFlux::inviscidFlux(qr, qgr, n);
    auto flux = vul::LDFSSFlux::inviscidFlux(ql, qr, qgl, qgr, n);
    REQUIRE(consistent_flux[0] == flux[0]);
    REQUIRE(consistent_flux[1] == flux[1]);
    REQUIRE(consistent_flux[2] == flux[2]);
    REQUIRE(consistent_flux[3] == flux[3]);
    REQUIRE(consistent_flux[4] == flux[4]);
  }
}

#include <catch.hpp>
#include <vul/Flux.h>
#include <vul/PhysicalFlux.h>
#include <vul/PerfectGas.h>

TEST_CASE("Physical flux tests"){

    vul::StaticArray<5> Q = {1.0, 0.1, 0.2, 0.3, 1.5};
    vul::Point<double> normal = {1, 0, 0};

    double rho = Q[0];
    double u = Q[1] / Q[0];
    double v = Q[2] / Q[0];
    double w = Q[3] / Q[0];
    double E = Q[4];
    double gamma = 1.4;

    double pressure = vul::perfect_gas::calcPressure(Q, gamma);
    double ubar = vul::perfect_gas::calcUBar(Q, normal);

    vul::StaticArray<2> QG = {gamma, pressure};

    auto flux = vul::PhysicalFlux::inviscidFlux(Q, QG, normal);
    REQUIRE(flux[0] == ubar * rho);
    REQUIRE(flux[1] == ubar * rho * u + normal.x * pressure);
    REQUIRE(flux[2] == ubar * rho * v + normal.y * pressure);
    REQUIRE(flux[3] == ubar * rho * w + normal.z * pressure);
    REQUIRE(flux[4] == ubar * (E + pressure));
}

TEST_CASE("can compute LDFSS flux") {
  vul::Point<double> n = {1.0/sqrt(3.0), 1.0/sqrt(3.0), 1.0/sqrt(3.0)};
  double gamma = 1.4;

  SECTION("constant state") {
    auto q = vul::StaticArray<5>{1.1, 0.1, 0.2, 0.3, 2.5};
    auto qg = vul::StaticArray<2>{gamma, vul::perfect_gas::calcPressure(q, gamma)};
    auto consistent_flux = vul::PhysicalFlux::inviscidFlux(q, qg, n);
    auto flux = vul::LDFSSFlux::inviscidFlux(q, q, qg, qg, n);
    REQUIRE(consistent_flux[0] == Approx(flux[0]).epsilon(1.0e-8));
    REQUIRE(consistent_flux[1] == Approx(flux[1]).epsilon(1.0e-8));
    REQUIRE(consistent_flux[2] == Approx(flux[2]).epsilon(1.0e-8));
    REQUIRE(consistent_flux[3] == Approx(flux[3]).epsilon(1.0e-8));
    REQUIRE(consistent_flux[4] == Approx(flux[4]).epsilon(1.0e-8));
  }
  SECTION("supersonic - upwind left state") {
    auto ql = vul::StaticArray<5>{1.1, 12.1, 12.2, 12.3, 355};
    auto qr = vul::StaticArray<5>{1.0, 10.1, 10.2, 10.3, 255};
    auto qgl = vul::StaticArray<2>{gamma, vul::perfect_gas::calcPressure(ql, gamma)};
    auto qgr = vul::StaticArray<2>{gamma, vul::perfect_gas::calcPressure(qr, gamma)};
    auto consistent_flux =
        vul::PhysicalFlux::inviscidFlux(ql, qgl, n);
    auto flux = vul::LDFSSFlux::inviscidFlux(ql, qr, qgl, qgr, n);
    REQUIRE(consistent_flux[0] == Approx(flux[0]).epsilon(1.0e-8));
    REQUIRE(consistent_flux[1] == Approx(flux[1]).epsilon(1.0e-8));
    REQUIRE(consistent_flux[2] == Approx(flux[2]).epsilon(1.0e-8));
    REQUIRE(consistent_flux[3] == Approx(flux[3]).epsilon(1.0e-8));
    REQUIRE(consistent_flux[4] == Approx(flux[4]).epsilon(1.0e-8));
  }
  SECTION("supersonic - upwind right state") {
    auto ql = vul::StaticArray<5>({1.0, -10.1, -10.2, -10.3, 200});
    auto qr = vul::StaticArray<5>({1.1, -12.1, -12.2, -12.3, 300});
    auto qgl = vul::StaticArray<2>{gamma, vul::perfect_gas::calcPressure(ql, gamma)};
    auto qgr = vul::StaticArray<2>{gamma, vul::perfect_gas::calcPressure(qr, gamma)};
    auto consistent_flux =
        vul::PhysicalFlux::inviscidFlux(qr, qgr, n);
    auto flux = vul::LDFSSFlux::inviscidFlux(ql, qr, qgl, qgr, n);
    REQUIRE(consistent_flux[0] == Approx(flux[0]).epsilon(1.0e-8));
    REQUIRE(consistent_flux[1] == Approx(flux[1]).epsilon(1.0e-8));
    REQUIRE(consistent_flux[2] == Approx(flux[2]).epsilon(1.0e-8));
    REQUIRE(consistent_flux[3] == Approx(flux[3]).epsilon(1.0e-8));
    REQUIRE(consistent_flux[4] == Approx(flux[4]).epsilon(1.0e-8));
  }
}

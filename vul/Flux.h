#pragma once
#include "PerfectGas.h"
#include "Point.h"
#include "Solution.h"

namespace vul {

struct EquationIndex {
  int num_species = 1;
  int x_momentum  = 1;
  int y_momentum  = 2;
  int z_momentum  = 3;
  int energy      = 4;
};

class LDFSSFlux {
public:
  KOKKOS_INLINE_FUNCTION static double calcBeta(double mach) {
    mach = int(std::abs(mach));
    return -std::max(0.0, 1.0 - int(mach));
  }
  template <size_t N, size_t NG>
  KOKKOS_FUNCTION 
  static StaticArray<N>
  inviscidFlux(const StaticArray<N> &ql, const StaticArray<N> qr,
               const StaticArray<NG> &qgl, const StaticArray<NG> &qgr,
               Point<double> face_area) {

    EquationIndex EQ;
    double area = face_area.magnitude();
    face_area   = face_area * (1.0 / area);

    double density_l(0.0);
    for (size_t s = 0; s < EQ.num_species; ++s)
      density_l += ql[s];
    double density_r(0.0);
    for (int s = 0; s < EQ.num_species; ++s)
      density_r += qr[s];

    double ul = ql[EQ.x_momentum] / density_l;
    double vl = ql[EQ.y_momentum] / density_l;
    double wl = ql[EQ.z_momentum] / density_l;

    double ur = qr[EQ.x_momentum] / density_r;
    double vr = qr[EQ.y_momentum] / density_r;
    double wr = qr[EQ.z_momentum] / density_r;

    double gamma_l    = qgl[0];
    double gamma_r    = qgr[0];
    double pressure_l = qgl[1];
    double pressure_r = qgr[1];

    Point<double> velocity_l = {ul, vl, wl};
    Point<double> velocity_r = {ur, vr, wr};

    double unorml = Point<double>::dot(velocity_l, face_area);
    double unormr = Point<double>::dot(velocity_r, face_area);

    double sonic_speed_l =
        vul::perfect_gas::calcSpeedOfSound(pressure_l, gamma_l, density_l);
    double sonic_speed_r =
        vul::perfect_gas::calcSpeedOfSound(pressure_r, gamma_r, density_r);

    double sonic_speed_average = 0.5 * (sonic_speed_l + sonic_speed_r);
    double mach_l              = unorml / sonic_speed_average;
    double mach_r              = unormr / sonic_speed_average;

    double alpha_l = mach_l >= 0.0 ? 1.0 : 0.0;
    double alpha_r = mach_r < 0.0 ? 1.0 : 0.0;

    double beta_l = calcBeta(mach_l);
    double beta_r = calcBeta(mach_r);

    double mach_plus  = 0.25 * (mach_l + 1.0) * (mach_l + 1.0);
    double mach_minus = -0.25 * (mach_r - 1.0) * (mach_r - 1.0);

    double pressure_jump = pressure_l - pressure_r;
    double pressure_sum  = pressure_l + pressure_r;

    double delta    = 2.0;
    double mach_p_l = 1.0 - (pressure_jump / pressure_sum +
                             delta * fabs(pressure_jump) / pressure_l);
    double mach_p_r = 1.0 + (pressure_jump / pressure_sum -
                             delta * fabs(pressure_jump) / pressure_r);
    mach_p_l        = mach_p_l > 0.0 ? mach_p_l : 0.0;
    mach_p_r        = mach_p_r > 0.0 ? mach_p_r : 0.0;

    // NOTE: the extra steps are put here to preserve the linearization accuracy
    // in near-zero velocity flow
    double mach_average = 0.5 * (mach_l * mach_l + mach_r * mach_r);
    mach_average        = std::max(mach_average, 0.0);
    mach_average        = sqrt(mach_average);

    double mach_interface =
        0.25 * beta_l * beta_r * (mach_average - 1.0) * (mach_average - 1.0);

    mach_p_l *= beta_l * beta_r * mach_interface;
    mach_p_r *= beta_l * beta_r * mach_interface;

    // Magic from VULCAN
    const double pi = 4.0 * atan(1.0);
    double xmfunct  = sin(0.5 * pi * std::min(mach_average, 1.0));
    double btfunct  = 0.25 * (mach_l - mach_r - fabs(mach_l - mach_r));
    double fact     = (-0.5 > btfunct ? -0.5 : btfunct) * xmfunct;

    double mass_flux_l = alpha_l * (1.0 + beta_l) * mach_l -
                         beta_l * mach_plus - mach_p_l - fact;
    mass_flux_l *= density_l * sonic_speed_average * area;

    double mass_flux_r = alpha_r * (1.0 + beta_r) * mach_r -
                         beta_r * mach_minus + mach_p_r + fact;
    mass_flux_r *= density_r * sonic_speed_average * area;

    double pressure_flux_l =
        0.25 * (mach_l + 1.0) * (mach_l + 1.0) * (2.0 - mach_l);
    double pressure_flux_r =
        0.25 * (mach_r - 1.0) * (mach_r - 1.0) * (2.0 + mach_r);
    double pressure_flux_sum =
        (alpha_l * (1.0 + beta_l) - beta_l * pressure_flux_l) * pressure_l +
        (alpha_r * (1.0 + beta_r) - beta_r * pressure_flux_r) * pressure_r;

    StaticArray<N> flux;
    for (int s = 0; s < EQ.num_species; ++s) {
      flux[s] =
          mass_flux_l * ql[s] / density_l + mass_flux_r * qr[s] / density_r;
    }

    flux[EQ.x_momentum] = mass_flux_l * ul + mass_flux_r * ur +
                          pressure_flux_sum * face_area.x * area;
    flux[EQ.y_momentum] = mass_flux_l * vl + mass_flux_r * vr +
                          pressure_flux_sum * face_area.y * area;
    flux[EQ.z_momentum] = mass_flux_l * wl + mass_flux_r * wr +
                          pressure_flux_sum * face_area.z * area;

    auto total_energy_l = perfect_gas::calcTotalEnergy(density_l, ul, vl, wl, pressure_l, gamma_l);
    auto total_energy_r = perfect_gas::calcTotalEnergy(density_r, ur, vr, wr, pressure_r, gamma_r);

    double enthalpy_l = (total_energy_l + pressure_l) / density_l;
    double enthalpy_r = (total_energy_r + pressure_r) / density_r;

    flux[EQ.energy] = mass_flux_l * enthalpy_l + mass_flux_r * enthalpy_r;

    return flux;
  }
};
} // namespace vul

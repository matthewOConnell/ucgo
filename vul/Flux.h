#pragma once
#include "Point.h"
#include "Solution.h"

namespace vul {

class LDFSSFlux {
public:
  inline static double calcBeta(double mach) {
    mach = int(std::abs(mach));
    return -std::max(0.0, 1.0 - int(mach));
  }
  template <size_t N>
  static StaticArray<N>
  inviscidFlux(const StaticArray<N> &ql, const StaticArray<N> qr,
               const StaticArray<3> &qgl, const StaticArray<3> &qgr,
               const Point<double> &area_normal) {
    Point<double> norm{area_normal.x, area_normal.y, area_normal.z};
    double area = norm.magnitude();
    norm        = norm * (1.0 / area);

    Point<double> u_l = {ql[1] / ql[0], ql[2] / ql[0], ql[3] / ql[0]};
    Point<double> u_r = {qr[1] / qr[0], qr[2] / qr[0], qr[3] / qr[0]};

    double unorml = u_l.dot(norm);
    double unormr = u_r.dot(norm);

    double gamma_l       = qgl[0];
    double gamma_r       = qgr[0];
    double press_l       = qgl[1]; // static pressure
    double press_r       = qgr[1]; // static pressure
    double rho_l         = ql[0];  // bulk density
    double rho_r         = qr[0];  // bulk density
    double sonic_speed_l = sqrt(press_l * gamma_l / rho_l);
    double sonic_speed_r = sqrt(press_r * gamma_r / rho_r);

    double sonic_speed_average = 0.5 * (sonic_speed_l + sonic_speed_r);
    double mach_left           = unorml / sonic_speed_average;
    double mach_right          = unormr / sonic_speed_average;

    double alpha_left  = mach_left >= 0.0 ? 1.0 : 0.0;
    double alpha_right = mach_right < 0.0 ? 1.0 : 0.0;

    double beta_left  = calcBeta(mach_left);
    double beta_right = calcBeta(mach_right);

    double mach_plus  = 0.25 * (mach_left + 1.0) * (mach_left + 1.0);
    double mach_minus = -0.25 * (mach_right - 1.0) * (mach_right - 1.0);

    double pressure_jump = press_l - press_r;
    double pressure_sum  = press_l + press_r;

    const double delta  = 2.0;
    const double pi     = 4.0 * std::atan(1.0);
    double mach_p_left  = 1.0 - (pressure_jump / pressure_sum +
                                delta * std::fabs(pressure_jump) / press_l);
    double mach_p_right = 1.0 + (pressure_jump / pressure_sum -
                                 delta * std::fabs(pressure_jump) / press_r);
    mach_p_left         = mach_p_left > 0.0 ? mach_p_left : 0.0;
    mach_p_right        = mach_p_right > 0.0 ? mach_p_right : 0.0;

    // NOTE: the extra steps are put here to preserve the linearization accuracy
    // in near-zero velocity flow
    double mach_average =
        0.5 * (mach_left * mach_left + mach_right * mach_right);
    mach_average = std::max(mach_average, 0.0);
    mach_average = sqrt(mach_average);

    double mach_interface = 0.25 * beta_left * beta_right *
                            (mach_average - 1.0) * (mach_average - 1.0);

    mach_p_left *= beta_left * beta_right * mach_interface;
    mach_p_right *= beta_left * beta_right * mach_interface;

    // Magic from VULCAN
    double xmfunct = sin(0.5 * pi * std::min(mach_average, 1.0));
    double btfunct =
        0.25 * (mach_left - mach_right - std::fabs(mach_left - mach_right));
    double fact = (-0.5 > btfunct ? -0.5 : btfunct) * xmfunct;

    //        double density_left(0.0);
    //        for (size_t species = 0; species < EQ::NumSpecies; ++species)
    //        density_left += ql[species]; double density_right(0.0); for
    //        (size_t species = 0; species < EQ::NumSpecies; ++species)
    //        density_right += qr[species];
    double density_left  = ql[0];
    double density_right = qr[0];

    double mass_flux_left = alpha_left * (1.0 + beta_left) * mach_left -
                            beta_left * mach_plus - mach_p_left - fact;
    mass_flux_left *= density_left * sonic_speed_average * area;

    double mass_flux_right = alpha_right * (1.0 + beta_right) * mach_right -
                             beta_right * mach_minus + mach_p_right + fact;
    mass_flux_right *= density_right * sonic_speed_average * area;

    double pressure_flux_left =
        0.25 * (mach_left + 1.0) * (mach_left + 1.0) * (2.0 - mach_left);
    double pressure_flux_right =
        0.25 * (mach_right - 1.0) * (mach_right - 1.0) * (2.0 + mach_right);
    double pressure_flux_sum =
        (alpha_left * (1.0 + beta_left) - beta_left * pressure_flux_left) *
            press_l +
        (alpha_right * (1.0 + beta_right) - beta_right * pressure_flux_right) *
            press_r;

    //        for (size_t species = 0; species < EQ::NumSpecies; ++species) {
    //            flux[species] = mass_flux_left * ql[species] / density_left +
    //            mass_flux_right * qr[species] / density_right;
    //        }

    StaticArray<N> flux;
    flux[0] = mass_flux_left + mass_flux_right;

    const auto &ul = ql[1];
    const auto &vl = ql[2];
    const auto &wl = ql[3];

    const auto &ur = qr[1];
    const auto &vr = qr[2];
    const auto &wr = qr[3];

    flux[1] = mass_flux_left * ul + mass_flux_right * ur +
              pressure_flux_sum * norm.x * area;
    flux[2] = mass_flux_left * vl + mass_flux_right * vr +
              pressure_flux_sum * norm.y * area;
    flux[3] = mass_flux_left * wl + mass_flux_right * wr +
              pressure_flux_sum * norm.z * area;

    double total_energy_l = press_l / (gamma_l - 1.0) + 0.5 * rho_l * (u_l.x * u_l.x + u_l.y * u_l.y + u_l.z * u_l.z);
    double total_energy_r = press_r / (gamma_l - 1.0) + 0.5 * rho_r * (u_r.x * u_r.x + u_r.y * u_r.y + u_r.z * u_r.z);

    double enthalpy_l = (total_energy_l + press_l) / density_left;
    double enthalpy_r = (total_energy_r + press_r) / density_right;

    flux[4] = mass_flux_left * enthalpy_l + mass_flux_right * enthalpy_r;
    return flux;
  }
};
} // namespace vul
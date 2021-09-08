#pragma once
#include <vul/Solution.h>
#include <vul/Point.h>
#include <math.h>

namespace vul {
namespace perfect_gas {
inline double calcSpeedOfSound(double pressure, double gamma, double density){
  return sqrt(pressure * gamma / density);
}
inline double calcPressure(const StaticArray<5> &q, double gamma) {
  return (gamma - 1.0) *
         (q[4] - 0.5 * (q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) / q[0]);
}
inline double calcTotalEnergy(double rho, double u, double v, double w, double P,
                       double gamma) {
  return P / (gamma - 1.0) + 0.5 * rho * (u * u + v * v + w * w);
}
inline double calcUBar(const StaticArray<5>&q, const Point<double>& norm){
  return (q[1] * norm.x + q[2] * norm.y + q[3] * norm.z) / q[0];
}
} // namespace perfect_gas
} // namespace vul

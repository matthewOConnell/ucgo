#pragma once
#include <math.h>
#include <Kokkos_Core.hpp>

namespace vul {
template <typename T>
class Point {
public:
  Point() = default;
  KOKKOS_FUNCTION Point(T x, T y, T z) : x(x), y(y), z(z){

  }
  KOKKOS_FUNCTION T magnitude() const {
    return sqrt(x*x + y*y + z*z);
  }
  KOKKOS_FUNCTION Point<T> normalized() const {
    auto m = magnitude();
    return Point<T>{x/m, y/m, z/m};
  }

  KOKKOS_FUNCTION static T dot(const Point<T>& v1, const Point<T>&v2){
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
  }
  KOKKOS_FUNCTION T dot(const Point<T>& v) const {
    return x*v.x + y*v.y + z*v.z;
  }
  KOKKOS_FUNCTION Point<T> cross(const Point<T>& b) const {
    return Point<T>(y * b.z - b.y*z, b.x * z - x * b.z, x * b.y - b.x * y);
  }

  KOKKOS_FUNCTION Point<T> operator-(const Point<T>& r) const {
    return Point<T>{x - r.x, y - r.y, z - r.z};
  }
  KOKKOS_FUNCTION Point<T> operator+(const Point<T>& r) const {
    return Point<T>{x + r.x, y + r.y, z + r.z};
  }
  KOKKOS_FUNCTION Point<T> operator*(const T& t) const {
    return Point<T>{x*t, y*t, z*t};
  }
public:
  union {
    struct {T x, y, z; };
    T pos[3];
  };
};
}

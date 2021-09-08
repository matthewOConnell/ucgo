#pragma once
#include <math.h>

namespace vul {
template <typename T>
class Point {
public:
  Point() = default;
  Point(T x, T y, T z) : x(x), y(y), z(z){

  }
  T magnitude() const {
    return sqrt(x*x + y*y + z*z);
  }
  Point<T> normalized() const {
    auto m = magnitude();
    return Point<T>{x/m, y/m, z/m};
  }

  static T dot(const Point<T>& v1, const Point<T>&v2){
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
  }
  T dot(const Point<T>& v) const {
    return x*v.x + y*v.y + z*v.z;
  }
  Point<T> cross(const Point<T>& b) const {
    return Point<T>(y * b.z - b.y*z, b.x * z - x * b.z, x * b.y - b.x * y);
  }

  Point<T> operator-(const Point<T>& r) const {
    return Point<T>{x - r.x, y - r.y, z - r.z};
  }
  Point<T> operator+(const Point<T>& r) const {
    return Point<T>{x + r.x, y + r.y, z + r.z};
  }
  Point<T> operator*(const T& t) const {
    return Point<T>{x*t, y*t, z*t};
  }
public:
  union {
    struct {T x, y, z; };
    T pos[3];
  };
};
}

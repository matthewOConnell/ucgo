//Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.
//Third Party Software:
//This software calls the following third party software, which is subject to the terms and conditions of its licensor, as applicable at the time of licensing.  Third party software is not bundled with this software, but may be available from the licensor.  License hyperlinks are provided here for information purposes only:  Kokkos v3.0, 3-clause BSD license, https://github.com/kokkos/kokkos, under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this third-party software.
//The Unstructured CFD graph operations miniapp platform is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. 
//Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
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

  KOKKOS_FUNCTION bool approxEqual(const Point<T>& rhs, double tol = 1.0e-12) const{
    auto diff = *this - rhs;
    return diff.magnitude() < tol;
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

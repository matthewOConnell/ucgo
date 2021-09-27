// Copyright 2021 United States Government as represented by the Administrator
// of the National Aeronautics and Space Administration. No copyright is claimed
// in the United States under Title 17, U.S. Code. All Other Rights Reserved.
// Third Party Software:
// This software calls the following third party software, which is subject to
// the terms and conditions of its licensor, as applicable at the time of
// licensing.  Third party software is not bundled with this software, but may
// be available from the licensor.  License hyperlinks are provided here for
// information purposes only:  Kokkos v3.0, 3-clause BSD license,
// https://github.com/kokkos/kokkos, under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this third-party
// software. The Unstructured CFD graph operations miniapp platform is licensed
// under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by
// applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
// OF ANY KIND, either express or implied. See the License for the specific
// language governing permissions and limitations under the License.
#pragma once
#include "Flux.h"
#include "Grid.h"
#include "Solution.h"

namespace vul {
template <size_t N, size_t NG> class Residual {
public:
  Residual(const Grid<vul::Device> grid) : grid(grid) {}

  void calc(const SolutionArray<N, Device::space> &Q,
            const GradientArray<N, Device::space> & Q_grad,
            const SolutionArray<2, Device::space> &QG,
            const GradientArray<2, Device::space>& QG_grad,
            SolutionArray<N, Device::space> &R) {

    int num_faces       = grid.face_to_cell.extent_int(0);
    auto face_to_cell   = grid.face_to_cell;
    auto face_areas     = grid.face_area;
    auto do_calculation = KOKKOS_CLASS_LAMBDA(int f) {
      int cell_l = face_to_cell(f, 0);
      int cell_r = face_to_cell(f, 1);
      StaticArray<N> ql, qr;
      StaticArray<NG> qgl, qgr;

      //--- second order extrapolation
      auto cell_centroid_l = grid.getCellCentroid(cell_l);
      auto cell_centroid_r = grid.getCellCentroid(cell_r);
      auto face_centroid = grid.getFaceCentroid(f);
      auto delta_l = face_centroid - cell_centroid_l;
      auto delta_r = face_centroid - cell_centroid_r;

      //-- unlimited 2nd order extrapolation
      for (int e = 0; e < N; e++) {
        ql[e] = Q(cell_l, e) + Q_grad(cell_l, e, 0) * delta_l.x + Q_grad(cell_l, e, 1)*delta_l.y + Q_grad(cell_l, e, 2) * delta_l.z;
      }
      for (int e = 0; e < N; e++) {
        qr[e] = Q(cell_r, e) + Q_grad(cell_r, e, 0) * delta_r.x + Q_grad(cell_r, e, 1)*delta_r.y + Q_grad(cell_r, e, 2) * delta_r.z;
      }

      for (int e = 0; e < NG; e++) {
        qgl[e] = QG(cell_l, e) + QG_grad(cell_l, e, 0) * delta_l.x + QG_grad(cell_l, e, 1) * delta_l.y + QG_grad(cell_l, e, 2)* delta_l.z;
      }
      for (int e = 0; e < NG; e++) {
        qgr[e] = QG(cell_r, e) + QG_grad(cell_r, e, 0) * delta_r.x + QG_grad(cell_r, e, 1) * delta_r.y + QG_grad(cell_r, e, 2)* delta_r.z;
      }

      Point<double> face_area;
      face_area.x = face_areas(f, 0);
      face_area.y = face_areas(f, 1);
      face_area.z = face_areas(f, 2);
      auto F      = LDFSSFlux::inviscidFlux(ql, qr, qgl, qgr, face_area);

      for (int e = 0; e < N; e++) {
        Kokkos::atomic_add(&R(cell_l, e), F[e]);
      }
      for (int e = 0; e < N; e++) {
        Kokkos::atomic_add(&R(cell_r, e), -F[e]);
      }
    };

    Kokkos::parallel_for("residual-calc", num_faces, do_calculation);
  }

private:
  const Grid<vul::Device> grid;
};
} // namespace vul

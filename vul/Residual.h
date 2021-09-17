//Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.
//Third Party Software:
//This software calls the following third party software, which is subject to the terms and conditions of its licensor, as applicable at the time of licensing.  Third party software is not bundled with this software, but may be available from the licensor.  License hyperlinks are provided here for information purposes only:  Kokkos v3.0, 3-clause BSD license, https://github.com/kokkos/kokkos, under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this third-party software.
//The Unstructured CFD graph operations miniapp platform is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. 
//Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
#pragma once
#include "Flux.h"
#include "Grid.h"
#include "Solution.h"

namespace vul {
template <size_t N, size_t NG> class Residual {
public:
  Residual(const Grid *grid) : grid(grid) {}

  void calc(const SolutionArray<N> &Q, const SolutionArray<2>& QG, SolutionArray<N> &R) {

    int num_faces = grid->face_to_cell.extent_int(0);
    auto face_to_cell = grid->face_to_cell.d_view;
    auto face_areas = grid->face_area.d_view;
    auto do_calculation = KOKKOS_CLASS_LAMBDA(int f) {
      int cell_l = face_to_cell(f, 0);
      int cell_r = face_to_cell(f, 1);
      StaticArray<N> ql, qr;
      for (int e = 0; e < N; e++) {
        ql[e] = Q.d_view(cell_l, e);
      }
      for (int e = 0; e < N; e++) {
        qr[e] = Q.d_view(cell_r, e);
      }

      StaticArray<NG> qgl, qgr;
      for(int e = 0; e < NG; e++){
        qgl[e] = QG.d_view(cell_l, e);
      }
      for(int e = 0; e < NG; e++){
        qgr[e] = QG.d_view(cell_r, e);
      }
      Point<double> face_area;
      face_area.x = face_areas(f, 0);
      face_area.y = face_areas(f, 1);
      face_area.z = face_areas(f, 2);
      auto F = LDFSSFlux::inviscidFlux(ql, qr, qgl, qgr, face_area);

//       some kind of mutex on R(cell_l) R(cell_r)
      for(int e = 0; e < N; e++){
        Kokkos::atomic_add(&R.d_view(cell_l, e), F[e]);
      }
      for(int e = 0; e < N; e++){
        Kokkos::atomic_add(&R.d_view(cell_r, e), -F[e]);
      }
    };

  Kokkos::parallel_for("residual-calc", num_faces, do_calculation);
  
  }

private:
  const Grid *grid;
};
} // namespace vul

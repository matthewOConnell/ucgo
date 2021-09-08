#pragma once
#include "Flux.h"
#include "Grid.h"
#include "Solution.h"

namespace vul {
template <int N> class Residual {
public:
  Residual(const Grid *grid) : grid(grid) {}

  void calc(const SolutionArray<N> &Q, SolutionArray<N> &R) {

    int num_faces = grid->face_to_cell.extent_int(0);
    for (int f = 0; f < num_faces; f++) {
      int cell_l = grid->face_to_cell.d_view(f, 0);
      int cell_r = grid->face_to_cell.d_view(f, 1);
      StaticArray<N> ql, qr;
      for (int e = 0; e < 5; e++) {
        ql[e] = Q.d_view(cell_l, e);
      }
      for (int e = 0; e < 5; e++) {
        qr[e] = Q.d_view(cell_r, e);
      }

      StaticArray<3> qgl, qgr;
      Point<double> face_area;
      face_area.x = grid->face_area.d_view(f, 0);
      face_area.y = grid->face_area.d_view(f, 1);
      face_area.z = grid->face_area.d_view(f, 2);
      auto F = LDFSSFlux::inviscidFlux(ql, qr, qgl, qgr, face_area);

//       some kind of mutex on R(cell_l) R(cell_r)
      for(int e = 0; e < N; e++){
        Kokkos::atomic_add(&R.d_view(cell_l, e), F[e]);
      }
      for(int e = 0; e < N; e++){
        Kokkos::atomic_add(&R.d_view(cell_r, e), -F[e]);
      }
    }
  }

private:
  const Grid *grid;
};
} // namespace vul

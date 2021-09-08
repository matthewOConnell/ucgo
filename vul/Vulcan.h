#pragma once

#include "Grid.h"
#include "Residual.h"
#include "Solution.h"
#include <string>

namespace vul {
template <int NumEqns> class Vulcan {
public:
  Vulcan(std::string filename)
      : grid(filename), residual(&grid), Q("solution", grid.numCells()),
        R("residual", grid.numCells()) {}
  void solve(int num_iterations) {
    for (int n = 0; n < num_iterations; n++) {
      residual.calc(Q, R);
      updateQ();
    }
  }
  void updateQ() const {
    auto update = KOKKOS_LAMBDA(int c) {
      for (int e = 0; e < NumEqns; e++) {
        Q.d_view(c, e) = Q.d_view(c, e) - dt * R.d_view(c, e);
      }
    };

    Kokkos::parallel_for(grid.numCells(), update);
  }

public:
  Grid grid;
  Residual<NumEqns> residual;
  SolutionArray<NumEqns> Q;
  SolutionArray<NumEqns> R;
  double dt = 1.0e-8;
};
}

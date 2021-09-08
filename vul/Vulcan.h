#pragma once

#include "Grid.h"
#include "Macros.h"
#include "Residual.h"
#include "Solution.h"
#include <string>
#include <tuple>

namespace vul {
template <size_t NumEqns, size_t NumGasVars> class Vulcan {
public:
  Vulcan(std::string filename)
      : grid(filename), residual(&grid), Q("solution", grid.numCells()),
        QG("gas-variables", grid.numCells()), R("residual", grid.numCells()) {
    setInitialConditions();
  }
  void solve(int num_iterations) {
    for (int n = 0; n < num_iterations; n++) {
      residual.calc(Q, R);
      double norm = calcNorm(R);
      printf("%d L2(R) = %e\n", n, norm);
      updateQ();
      setBCs();
      calcGasVariables();
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

  void writeCSV(std::string filename) {
    Kokkos::deep_copy(Q.h_view, Q.d_view);
    FILE *fp = fopen(filename.c_str(), "w");
    VUL_ASSERT(fp != nullptr, "Could not open file for writing: " + filename);
    if (NumEqns == 5) {
      fprintf(fp, "density, u, v, w, E\n");
    } else {
      for (int e = 0; e < NumEqns; e++) {
        fprintf(fp, "Q_%d, ", e);
      }
      fprintf(fp, "\n");
    }
    for (int c = 0; c < grid.numCells(); c++) {
      for (int e = 0; e < NumEqns; e++) {
        fprintf(fp, "%e ", Q.h_view(c, e));
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  }

public:
  Grid grid;
  Residual<NumEqns, NumGasVars> residual;
  StaticArray<NumEqns> Q_reference;
  SolutionArray<NumEqns> Q;
  SolutionArray<NumEqns> R;
  SolutionArray<2> QG;
  double dt = 1.0e-8;

  void setInitialConditions() {
    Q_reference[0] = 1.000000;
    Q_reference[1] = 1.000000;
    Q_reference[2] = 0.000000;
    Q_reference[3] = 0.000000;
    Q_reference[4] = 2.255499;

    auto update = KOKKOS_LAMBDA(int c) {
      for (int e = 0; e < NumEqns; e++) {
        Q.d_view(c, e) = Q_reference[e];
      }
    };

    Kokkos::parallel_for(grid.numCells(), update);
  }
  void setBCs() {
    auto set = KOKKOS_LAMBDA(int c) {
      auto [type, index] = grid.cellIdToTypeAndIndexPair(c);
      if (type == vul::TRI or type == vul::QUAD) {
        for (int e = 0; e < NumEqns; e++) {
          Q.d_view(c, e) = Q_reference[e];
        }
      }
    };

    Kokkos::parallel_for(grid.numCells(), set);
  }

  double calcNorm(const SolutionArray<NumEqns> &A) {
    double norm    = 0.0;
    auto calc_norm = KOKKOS_LAMBDA(int c, double &norm) {
      for (int e = 0; e < NumEqns; e++) {
        norm += A.d_view(c, e) * A.d_view(c, e);
      }
    };

    Kokkos::parallel_reduce(grid.numVolumeCells(), calc_norm, norm);
    norm = sqrt(norm);
    return norm;
  }

  void calcGasVariables() {
    auto calc = KOKKOS_LAMBDA(int c) {
      QG.d_view(c, 0) = 1.4;
      auto q = Q.d_view;
      double press = (q(c, 0) - 1.0) * (q(c, 4) - 0.5 * (q(c, 1) * q(c, 1) + q(c, 2) * q(c, 2) + q(c, 3) * q(c, 3)) / q(c, 0));
      QG.d_view(c, 1) = press;
    };

    Kokkos::parallel_for(grid.numCells(), calc);
  }
};
} // namespace vul
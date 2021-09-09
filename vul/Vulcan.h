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
      residual.calc(Q, QG, R);
      double norm = calcNorm(R);
      printf("%d L2(R) = %e\n", n, norm);
      updateQ();
      setBCs();
      calcGasVariables();

      if(n % plot_freq == 0){
        Kokkos::deep_copy(Q.h_view, Q.d_view);
        Kokkos::deep_copy(QG.h_view, QG.d_view);
        writeCSV("output." + std::to_string(n) + ".csv");
      }
    }
  }
  void updateQ() const {
    auto update = KOKKOS_CLASS_LAMBDA(int c) {
      for (int e = 0; e < NumEqns; e++) {
        Q.d_view(c, e) = Q.d_view(c, e) - dt * R.d_view(c, e);
        R.d_view(c, e) = 0.0;
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
    for (int inf_id = 0; inf_id < grid.numCells(); inf_id++) {
      int vul_id = grid.getVulCellIdFromInfId(inf_id);
      for (int e = 0; e < NumEqns; e++) {
        fprintf(fp, "%e, ", Q.h_view(vul_id, e));
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
  SolutionArray<NumGasVars> QG;
  double dt = 10.0;
  int plot_freq = 10;

  void setInitialConditions() {
    Q_reference[0] = 1.000000;
    Q_reference[1] = 1.000000;
    Q_reference[2] = 0.000000;
    Q_reference[3] = 0.000000;
    Q_reference[4] = 2.255499;

    auto update = KOKKOS_CLASS_LAMBDA(int c) {
      for (int e = 0; e < NumEqns; e++) {
        Q.d_view(c, e) = Q_reference[e];
      }
      auto [type, index] = grid.cellIdToTypeAndIndexPair(c);
      if (type != vul::TRI and type != vul::QUAD) {
          Q.d_view(c, 1) = 1.1*Q_reference[1];
      }
    };

    Kokkos::parallel_for(grid.numCells(), update);

    setBCs();
    calcGasVariables();
  }
  void setBCs() {
    auto set = KOKKOS_CLASS_LAMBDA(int c) {
      auto [type, index] = grid.cellIdToTypeAndIndexPair(c);
      if (type == vul::TRI or type == vul::QUAD) {
        for (int e = 0; e < NumEqns; e++) {
          Q.d_view(c, e) = Q_reference[e];
        }
      }
    };

    Kokkos::parallel_for(grid.numCells(), set);
  }

  double calcNorm(const SolutionArray<NumEqns> &A_in) {
    double norm    = 0.0;
    auto A = A_in.d_view;
    auto calc_norm = KOKKOS_LAMBDA(int c, double &norm) {
      for (int e = 0; e < NumEqns; e++) {
        norm += A(c, e) * A(c, e);
      }
    };

    Kokkos::parallel_reduce(grid.numVolumeCells(), calc_norm, norm);
    norm = sqrt(norm);
    return norm;
  }

  void calcGasVariables() {
    auto calc = KOKKOS_CLASS_LAMBDA(int c) {
      QG.d_view(c, 0) = 1.4;
      StaticArray<NumEqns> q;
      for(int e = 0; e < NumEqns; e++){
        q[e] = Q.d_view(c, e);
      }
      double press = perfect_gas::calcPressure(q, QG.d_view(c, 0));
      QG.d_view(c, 1) = press;
    };

    Kokkos::parallel_for(grid.numCells(), calc);
    Kokkos::deep_copy(QG.h_view, QG.d_view);
  }
};
} // namespace vul

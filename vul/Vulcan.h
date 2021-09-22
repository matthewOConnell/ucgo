//Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.
//Third Party Software:
//This software calls the following third party software, which is subject to the terms and conditions of its licensor, as applicable at the time of licensing.  Third party software is not bundled with this software, but may be available from the licensor.  License hyperlinks are provided here for information purposes only:  Kokkos v3.0, 3-clause BSD license, https://github.com/kokkos/kokkos, under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this third-party software.
//The Unstructured CFD graph operations miniapp platform is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. 
//Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
#pragma once

#include "Grid.hpp"
#include "Macros.h"
#include "Residual.h"
#include "Solution.h"
#include <string>
#include <tuple>

namespace vul {
template <size_t NumEqns, size_t NumGasVars> class Vulcan {
public:
  Vulcan(const vul::Grid<vul::Host>& grid_host_i, const vul::Grid<vul::Device>& grid_device_i)
      : grid_host(grid_host_i), grid_device(grid_device_i), residual(&grid_device), Q("solution", grid_host.numCells()),
        QG("gas-variables", grid_host.numCells()), R("residual", grid_host.numCells()) {
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

      if(plot_freq > 0 and n % plot_freq == 0){
        syncToHost();
        writeCSV("output." + std::to_string(n) + ".csv");
      }
    }
  }

  void syncToHost() {
    Kokkos::deep_copy(Q.h_view, Q.d_view);
    Kokkos::deep_copy(QG.h_view, QG.d_view);
    Kokkos::deep_copy(R.h_view, R.d_view);
  }
  void updateQ() const {
    auto Q_device = Q.d_view;
    auto R_device = R.d_view;
    auto dt_device = dt;
    auto update = KOKKOS_LAMBDA(int c) {
      for (int e = 0; e < NumEqns; e++) {
        Q_device(c, e) = Q_device(c, e) - dt_device * R_device(c, e);
        R_device(c, e) = 0.0;
      }
    };

    Kokkos::parallel_for("updateQ", grid_host.numCells(), update);
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
    for (int inf_id = 0; inf_id < grid_host.numCells(); inf_id++) {
      int vul_id = grid_host.getVulCellIdFromInfId(inf_id);
      for (int e = 0; e < NumEqns; e++) {
        fprintf(fp, "%e, ", Q.h_view(vul_id, e));
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  }

public:
  Grid<vul::Host> grid_host;
  Grid<vul::Device> grid_device;
  Residual<NumEqns, NumGasVars> residual;
  StaticArray<NumEqns> Q_reference;
  SolutionArray<NumEqns> Q;
  SolutionArray<NumEqns> R;
  SolutionArray<NumGasVars> QG;
  double dt = 10.0;
  int plot_freq = -1;

  void setInitialConditions() {
      Kokkos::Profiling::pushRegion("setInitialConditions");
    Q_reference[0] = 1.000000;
    Q_reference[1] = 1.000000;
    Q_reference[2] = 0.000000;
    Q_reference[3] = 0.000000;
    Q_reference[4] = 2.255499;

    auto Q_device = Q.d_view;
    auto Q_ref = Q_reference;

    auto update_boundary = KOKKOS_LAMBDA(int c) {
      for (int e = 0; e < NumEqns; e++) {
        Q_device(c, e) = Q_ref[e];
      }
    };
    Kokkos::parallel_for("set initial condition boundary", Kokkos::RangePolicy<>(grid_host.boundaryCellsStart(), grid_host.boundaryCellsEnd()), update_boundary);

    auto update_interior = KOKKOS_LAMBDA(int c) {
      for (int e = 0; e < NumEqns; e++) {
          Q_device(c, e) = Q_ref[e];
      }
      Q_device(c, 1) = 1.1*Q_ref[1]; // perturb first density to give non zero res
    };
    Kokkos::parallel_for("set initial condition interior", grid_host.numVolumeCells(), update_interior);

    Kokkos::Profiling::popRegion();

    setBCs();
    calcGasVariables();
  }
  void setBCs() {

    auto Q_device = Q.d_view;
    auto Q_ref = Q_reference; // make a local copy to be transferred to the GPU
    int boundary_cell_start = grid_host.boundaryCellsStart();
    int boundary_cell_end = grid_host.boundaryCellsEnd();
    auto set = KOKKOS_LAMBDA(int c) {
        for (int e = 0; e < NumEqns; e++) {
          Q_device(c, e) = Q_ref[e];
        }
    };

    Kokkos::parallel_for("setBCs", Kokkos::RangePolicy<>(boundary_cell_start, boundary_cell_end), set);
  }

  double calcNorm(const SolutionArray<NumEqns> &A_in) {
    double norm    = 0.0;
    auto A = A_in.d_view;
    auto calc_norm = KOKKOS_LAMBDA(int c, double &norm) {
      for (int e = 0; e < NumEqns; e++) {
        norm += A(c, e) * A(c, e);
      }
    };

    Kokkos::parallel_reduce("calc R norm", grid_host.numVolumeCells(), calc_norm, norm);
    norm = sqrt(norm);
    return norm;
  }

  void calcGasVariables() {
    auto QG_device = QG.d_view;
    auto Q_device = Q.d_view;
    auto calc = KOKKOS_LAMBDA(int c) {
      QG_device(c, 0) = 1.4;
      StaticArray<NumEqns> q;
      for(int e = 0; e < NumEqns; e++){
        q[e] = Q_device(c, e);
      }
      double press = perfect_gas::calcPressure(q, QG_device(c, 0));
      QG_device(c, 1) = press;
    };

    Kokkos::parallel_for("calcGasVariables", grid_host.numCells(), calc);
  }
};
} // namespace vul

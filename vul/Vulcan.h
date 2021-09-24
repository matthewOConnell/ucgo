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

#include "Gradients.h"
#include "Grid.hpp"
#include "Macros.h"
#include "Residual.h"
#include "Solution.h"
#include <string>
#include <tuple>

namespace vul {
template <size_t NumEqns, size_t NumGasVars> class Vulcan {
public:
  Vulcan(const vul::Grid<vul::Host> &grid_host_i,
         const vul::Grid<vul::Device> &grid_device_i)
      : grid_host(grid_host_i), grid_device(grid_device_i),
        unweighted_least_squares(grid_host), residual(&grid_device),
        Q_device("solution", grid_host.numCells()),
        Q_host("solution", grid_host.numCells()),
        R("residual", grid_host.numCells()),
        QG_host("gas-variables", grid_host.numCells()),
        QG_device("gas-variables", grid_host.numCells()),
        Q_grad_nodes("grad-Q-nodes", grid_host.numCells()),
        QG_grad_nodes("grad-GQ-nodes", grid_host.numCells()),
        Q_grad_faces("grad-Q-faces", grid_host.numFaces()),
        QG_grad_faces("grad-QG-faces", grid_host.numFaces()) {
    setInitialConditions();
  }
  void calcGradients() {
    unweighted_least_squares.calcMultipleGrads<NumEqns>(Q_device, grid_device,
                                                        Q_grad_nodes);
    unweighted_least_squares.calcMultipleGrads<NumGasVars>(
        QG_device, grid_device, QG_grad_nodes);

    averageNodeToFace<NumEqns, 3>(Q_grad_nodes, Q_grad_faces);
    averageNodeToFace<NumGasVars, 3>(QG_grad_nodes, QG_grad_faces);
  }

  template <size_t N, size_t D>
  void averageNodeToFace(Kokkos::View<double *[N][D]> field,
                         Kokkos::View<double *[N][D]> face_field) {
    auto grid = grid_device;
    auto average = KOKKOS_LAMBDA(int f) {
      bool is_quad = grid.face_to_nodes(f, 3) != -1;
      for (int e = 0; e < N; e++) {
        for (int d = 0; d < D; d++) {
          double running_tally = 0.0;
          for (int i = 0; i < 3; i++) {
            int node = grid.face_to_nodes(f, i);
            running_tally += field(node, e, d);
          }
          if (is_quad) {
            int node = grid.face_to_nodes(f, 3);
            running_tally += field(node, e, d);
            running_tally *= 0.25;
          } else {
            running_tally /= 3.0;
          }
          face_field(f, e, d) = running_tally;
        }
      }
    };

    Kokkos::parallel_for("node-to-face-avg", grid_host.face_to_nodes.extent(0),
                         average);
  }
  void solve(int num_iterations) {
    for (int n = 0; n < num_iterations; n++) {
      calcGradients();
      residual.calc(Q_device, Q_grad_nodes, QG_device, QG_grad_nodes, R);
      double norm = calcNorm(R);
      printf("%d L2(R) = %e\n", n, norm);
      updateQ();
      setBCs();
      calcGasVariables();

      if (plot_freq > 0 and n % plot_freq == 0) {
        syncToHost();
        writeCSV("output." + std::to_string(n) + ".csv");
      }
    }
  }

  void syncToHost() {
    vul::force_copy(Q_host, Q_host);
    vul::force_copy(QG_host, QG_host);
  }
  void updateQ() const {
    auto dt_device = dt;
    auto Q_d = Q_device;
    auto R_d = R;
    auto update    = KOKKOS_LAMBDA(int c) {
      for (int e = 0; e < NumEqns; e++) {
        Q_d(c, e) = Q_d(c, e) - dt_device * R_d(c, e);
        R_d(c, e) = 0.0;
      }
    };

    Kokkos::parallel_for("updateQ", grid_host.numCells(), update);
  }

  void writeCSV(std::string filename) {
    vul::force_copy(Q_host, Q_device);
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
        fprintf(fp, "%e, ", Q_host(vul_id, e));
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  }

public:
  Grid<vul::Host> grid_host;
  Grid<vul::Device> grid_device;
  LeastSquares unweighted_least_squares;
  Residual<NumEqns, NumGasVars> residual;
  StaticArray<NumEqns> Q_reference;
  SolutionArray<NumEqns, vul::Device::space> Q_device;
  SolutionArray<NumGasVars, vul::Device::space> QG_device;
  SolutionArray<NumEqns, vul::Host::space> Q_host;
  SolutionArray<NumGasVars, vul::Host::space> QG_host;

  GradientArray<NumEqns, vul::Device::space> Q_grad_nodes;
  GradientArray<NumGasVars, vul::Device::space> QG_grad_nodes;
  GradientArray<NumEqns, vul::Device::space> Q_grad_faces;
  GradientArray<NumGasVars, vul::Device::space> QG_grad_faces;

  SolutionArray<NumEqns, vul::Device::space> R;
  double dt     = 10.0;
  int plot_freq = -1;

  void setInitialConditions() {
    Kokkos::Profiling::pushRegion("setInitialConditions");
    Q_reference[0] = 1.000000;
    Q_reference[1] = 1.000000;
    Q_reference[2] = 0.000000;
    Q_reference[3] = 0.000000;
    Q_reference[4] = 2.255499;

    auto Q_ref    = Q_reference;

    auto Q_d = Q_device;
    auto update_boundary = KOKKOS_LAMBDA(int c) {
      for (int e = 0; e < NumEqns; e++) {
        Q_d(c, e) = Q_ref[e];
      }
    };
    Kokkos::parallel_for("set initial condition boundary",
                         Kokkos::RangePolicy<>(grid_host.boundaryCellsStart(),
                                               grid_host.boundaryCellsEnd()),
                         update_boundary);

    auto update_interior = KOKKOS_LAMBDA(int c) {
      for (int e = 0; e < NumEqns; e++) {
        Q_d(c, e) = Q_ref[e];
      }
      Q_d(c, 1) =
          1.1 * Q_ref[1]; // perturb first density to give non zero res
    };
    Kokkos::parallel_for("set initial condition interior",
                         grid_host.numVolumeCells(), update_interior);
    Kokkos::Profiling::popRegion();

    setBCs();
    calcGasVariables();
  }
  void setBCs() {
    auto Q_ref = Q_reference; // make a local copy to be transferred to the GPU
    int boundary_cell_start = grid_host.boundaryCellsStart();
    int boundary_cell_end   = grid_host.boundaryCellsEnd();
    auto Q_d = Q_device;
    auto set                = KOKKOS_LAMBDA(int c) {
      for (int e = 0; e < NumEqns; e++) {
        Q_d(c, e) = Q_ref[e];
      }
    };

    Kokkos::parallel_for(
        "setBCs", Kokkos::RangePolicy<>(boundary_cell_start, boundary_cell_end),
        set);
  }

  double calcNorm(const SolutionArray<NumEqns, vul::Device::space> &A) {
    double norm    = 0.0;
    auto calc_norm = KOKKOS_LAMBDA(int c, double &norm) {
      for (int e = 0; e < NumEqns; e++) {
        norm += A(c, e) * A(c, e);
      }
    };

    Kokkos::parallel_reduce("calc R norm", grid_host.numVolumeCells(),
                            calc_norm, norm);
    norm = sqrt(norm);
    return norm;
  }

  void calcGasVariables() {
    auto QG_d = QG_device;
    auto Q_d = Q_device;
    auto calc      = KOKKOS_LAMBDA(int c) {
      QG_d(c, 0) = 1.4;
      StaticArray<NumEqns> q;
      for (int e = 0; e < NumEqns; e++) {
        q[e] = Q_d(c, e);
      }
      double press    = perfect_gas::calcPressure(q, QG_d(c, 0));
      QG_d(c, 1) = press;
    };

    Kokkos::parallel_for("calcGasVariables", grid_host.numCells(), calc);
  }
};
} // namespace vul

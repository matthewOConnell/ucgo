#pragma once
#include "Decompositions.h"
#include "Grid.h"
#include "Macros.h"

namespace vul {
template <typename getPoint, typename getWeight, typename Point, typename Row>
void setLSQWeights(getPoint get_neighbor_point, Row row,
<<<<<<< HEAD
                   getWeight get_neighbor_weight, const Point &center_point,
                   Kokkos::View<double *[3], vul::Host::space> coeffs_write,
                   long write_offset) {
=======
                   getWeight get_neighbor_weight, const Point &center_point, Kokkos::View<double*[3], vul::Host::space> coeffs_write, long write_offset) {
>>>>>>> l-value-copy
  using Matrix = vul::DynamicMatrix<double>;
  Matrix A(row.size(), 4);
  for (int i = 0; i < row.size(); ++i) {
    auto neighbor = row(i);
    auto w        = get_neighbor_weight(neighbor);
    auto dist     = get_neighbor_point(neighbor) - center_point;
    //      auto p = get_neighbor_point(neighbor);
    A(i, 0) = w * 1.0;
    A(i, 1) = w * dist.x;
    A(i, 2) = w * dist.y;
    A(i, 3) = w * dist.z;
  }
  Matrix Q, R;
  std::tie(Q, R) = vul::householderDecomposition(A);
  auto Ainv      = vul::calcPseudoInverse(Q, R);
  VUL_ASSERT(Ainv.columns == row.size(),
             "Ainv should be same size (" + std::to_string(Ainv.rows) +
                 ") as rows" + std::to_string(row.size()));
  for (int i = 0; i < row.size(); ++i) {
    auto neighbor = row(i);
<<<<<<< HEAD
    auto w        = get_neighbor_weight(neighbor);
=======
    auto w           = get_neighbor_weight(neighbor);
>>>>>>> l-value-copy
    // coeffs_write(write_offset + i, 0) = w * Ainv(0, i);
    coeffs_write(write_offset + i, 0) = w * Ainv(1, i);
    coeffs_write(write_offset + i, 1) = w * Ainv(2, i);
    coeffs_write(write_offset + i, 2) = w * Ainv(3, i);
  }
}
class LeastSquares {
public:
  LeastSquares(const vul::Grid<vul::Host> &grid)
<<<<<<< HEAD
      : coeffs(NoInit("lsq_coeffs"), grid.node_to_cell.num_non_zero) {
    auto coeffs_host = Kokkos::View<double *[3], vul::Host::space>(
        NoInit("lsq-host-coeffs"), grid.node_to_cell.num_non_zero);
=======
      : coeffs("lsq_coeffs", grid.node_to_cell.num_non_zero) {
    auto coeffs_host =  Kokkos::View<double* [3], vul::Host::space>("lsq-host-coeffs", grid.node_to_cell.num_non_zero);
>>>>>>> l-value-copy
    auto cell_centroids  = grid.cell_centroids;
    auto getCellCentroid = [&](int cell) {
      vul::Point<double> p;
      p.x = cell_centroids(cell, 0);
      p.y = cell_centroids(cell, 1);
      p.z = cell_centroids(cell, 2);
      return p;
    };

    auto points   = grid.points;
    auto getPoint = [&](int node) {
      vul::Point<double> p;
      p.x = points(node, 0);
      p.y = points(node, 1);
      p.z = points(node, 2);
      return p;
    };

    Kokkos::Profiling::pushRegion("setLSQWeights");
    Kokkos::parallel_for(
        "calc LSQ Weights", HostPolicy(0, grid.node_to_cell.num_rows),
        [&](int node) {
          auto get_neighbor_weight = [&](int i) { return 1.0; };
          setLSQWeights(getCellCentroid, grid.node_to_cell(node),
                        get_neighbor_weight, getPoint(node), coeffs_host,
                        grid.node_to_cell.rowStart(node));
        });
    Kokkos::Profiling::popRegion();
    vul::force_copy(coeffs, coeffs_host);
  }

public:
  Kokkos::View<double *[3], vul::Device::space> coeffs;

  template <size_t N>
  void calcMultipleGrads(Kokkos::View<double *[N]> fields,
                         const vul::Grid<vul::Device> &grid,
                         Kokkos::View<double *[N][3]> grad) {

    long num_nodes = grid.node_to_cell.num_rows;
    Kokkos::deep_copy(grad, 0.0);

    auto n2c = grid.node_to_cell;
    auto calc_node_grad = KOKKOS_CLASS_LAMBDA(int n, int e, int dir) {
      auto start = n2c.rowStart(n);
      auto end = n2c.rowEnd(n);
      for(int index = start; index < end; index++){
        long neighbor = n2c.cols(index);
        double d      = fields(neighbor, e);
        grad(n, e, dir) += coeffs(index, dir) * d;
      }
    };
    Kokkos::parallel_for(
        "calc grad",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {num_nodes, N, 3}),
        calc_node_grad);
  }

  template <typename GetFieldValue>
  void calcGrad(GetFieldValue get_field_value,
                const vul::Grid<vul::Device> &grid,
                Kokkos::View<double *[3]> grad) const {
    long num_nodes    = grid.node_to_cell.num_rows;
    for (int c = 0; c < num_nodes; c++) {
      grad(c, 0) = 0.0;
      grad(c, 1) = 0.0;
      grad(c, 2) = 0.0;
    }

    auto n2c            = grid.node_to_cell;
    auto calc_node_grad = KOKKOS_CLASS_LAMBDA(int n) {
      auto row = n2c(n);
      for (int i = 0; i < row.size(); i++) {
        int index     = row.row_index_start + i;
        long neighbor = row(i);
        double d      = get_field_value(neighbor);
        grad(n, 0) += coeffs(index, 0) * d;
        grad(n, 1) += coeffs(index, 1) * d;
        grad(n, 2) += coeffs(index, 2) * d;
      }
    };
    Kokkos::parallel_for("calc grad", num_nodes, calc_node_grad);
  }
};

} // namespace vul

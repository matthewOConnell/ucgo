#pragma once
#include "Decompositions.h"
#include "Grid.h"
#include "Macros.h"

namespace vul {
template <typename getPoint, typename getWeight, typename Point, typename Row>
void setLSQWeights(getPoint get_neighbor_point, Row row,
                   getWeight get_neighbor_weight, const Point &center_point,
                   Kokkos::View<double *[3], vul::Host::space> coeffs_write,
                   long write_offset) {
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
    auto w        = get_neighbor_weight(neighbor);
    // coeffs_write(write_offset + i, 0) = w * Ainv(0, i);
    coeffs_write(write_offset + i, 0) = w * Ainv(1, i);
    coeffs_write(write_offset + i, 1) = w * Ainv(2, i);
    coeffs_write(write_offset + i, 2) = w * Ainv(3, i);
  }
}
class LeastSquares {
public:
  LeastSquares(const vul::Grid<vul::Host> &grid)
      : coeffs(NoInit("lsq_coeffs"), grid.node_to_cell.num_non_zero),
        coeffs_transpose(NoInit("lsq_coeffs_transpose"),
                         grid.cell_to_node.num_non_zero),
        cell_ids_in_crs_ordering(NoInit("cell_ids_from_crs_index"),
                                 grid.cell_to_node.num_non_zero) {
    auto coeffs_host = Kokkos::View<double *[3], vul::Host::space>(
        NoInit("lsq-host-coeffs"), grid.node_to_cell.num_non_zero);
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

    copyCoeffsToTranspose(grid, coeffs_host);
    setCellIdsFromCRSIndex(grid);
  }

  void setCellIdsFromCRSIndex(const Grid<Host> &grid_host) {
    Kokkos::View<long *, Host::space> cell_ids_in_crs_ordering_host(
        "cell_ids_in_crs_order_host", grid_host.cell_to_node.num_non_zero);
    auto set_cell_ids = KOKKOS_LAMBDA(int cell) {
      auto start = grid_host.cell_to_node.rowStart(cell);
      auto end   = grid_host.cell_to_node.rowEnd(cell);
      for (auto index = start; index < end; index++) {
        cell_ids_in_crs_ordering_host(index) = cell;
      }
    };
    Kokkos::parallel_for("set_cell_ids_from_crs",
                         HostPolicy(0, grid_host.numCells()), set_cell_ids);
    vul::force_copy(cell_ids_in_crs_ordering, cell_ids_in_crs_ordering_host);
  }

  void copyCoeffsToTranspose(
      const vul::Grid<vul::Host> &grid_host,
      Kokkos::View<double *[3], vul::Host::space> coeffs_host) {

    auto coeffs_transpose_host = Kokkos::View<double *[3], vul::Host::space>(
        NoInit("lsq-host-coeffs-transpose"),
        grid_host.cell_to_node.num_non_zero);
    auto do_transpose = KOKKOS_LAMBDA(int node) {
      auto n2c_start = grid_host.node_to_cell.rowStart(node);
      auto n2c_end   = grid_host.node_to_cell.rowEnd(node);
      for (int n2c_index = n2c_start; n2c_index < n2c_end; n2c_index++) {
        int cell       = grid_host.node_to_cell.cols(n2c_index);
        auto c2n_start = grid_host.cell_to_node.rowStart(cell);
        auto c2n_end   = grid_host.cell_to_node.rowEnd(cell);
        for (int c2n_index = c2n_start; c2n_index < c2n_end; c2n_index++) {
          int other_node = grid_host.cell_to_node.cols(c2n_index);
          if (other_node == node) {
            coeffs_transpose_host(c2n_index, 0) = coeffs_host(n2c_index, 0);
            coeffs_transpose_host(c2n_index, 1) = coeffs_host(n2c_index, 1);
            coeffs_transpose_host(c2n_index, 2) = coeffs_host(n2c_index, 2);
          }
        }
      }
    };

    Kokkos::parallel_for("lsq copy coeff transpose",
                         HostPolicy(0, grid_host.numPoints()), do_transpose);
    vul::force_copy(coeffs_transpose, coeffs_transpose_host);
  }

public:
  Kokkos::View<double *[3], vul::Device::space> coeffs;
  Kokkos::View<double *[3], vul::Device::space> coeffs_transpose;
  Kokkos::View<long *, vul::Device::space> cell_ids_in_crs_ordering;

  void printSummary(Kokkos::View<double *[3]> coeffs_function,
                    CompressedRowGraph<vul::Host> &graph) const {
    auto first_few = std::min(long(3), graph.num_rows);
    for (int row = 0; row < first_few; row++) {
      auto start = graph.rowStart(row);
      auto end   = graph.rowEnd(row);
      printf("Row %d: Coeffs: ", row);
      for (int index = start; index < end; index++) {
        printf(" %e", coeffs_function(index, 0));
      }
      printf("\n");
    }
  }

  template <size_t N>
  void calcMultipleGrads_transpose(Kokkos::View<double *[N]> fields,
                                   const vul::Grid<vul::Device> &grid,
                                   Kokkos::View<double *[N][3]> grad) {

    long num_cells = grid.cell_to_node.num_rows;
    Kokkos::deep_copy(grad, 0.0);

    auto c2n            = grid.cell_to_node;
    auto calc_node_grad = KOKKOS_CLASS_LAMBDA(int cell, int equation, int dir) {
      auto start = c2n.rowStart(cell);
      auto end   = c2n.rowEnd(cell);
      double d   = fields(cell, equation);
      for (int index = start; index < end; index++) {
        long node = c2n.cols(index);
        Kokkos::atomic_add(&grad(node, equation, dir),
                           coeffs_transpose(index, dir) * d);
      }
    };
    Kokkos::parallel_for(
        "calc_grad_cell",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {num_cells, N, 3}),
        calc_node_grad);
  }

  template <size_t N>
  void calcMultipleGrads_flat(Kokkos::View<double *[N]> fields,
                              const vul::Grid<vul::Device> &grid,
                              Kokkos::View<double *[N][3]> grad) {

    long num_non_zeros = grid.cell_to_node.num_non_zero;
    Kokkos::deep_copy(grad, 0.0);

    auto c2n = grid.cell_to_node;
    auto calc_node_grad =
        KOKKOS_CLASS_LAMBDA(int work_item) {
          long index = work_item % num_non_zeros;
          long equation = work_item / num_non_zeros;
      auto cell = cell_ids_in_crs_ordering(index);
      auto node = c2n.cols(index);
      double d  = fields(cell, equation);
      for(int dir = 0; dir < 3; dir++){
      grad(node, equation, dir) += coeffs_transpose(index, dir)* d;
      // Kokkos::atomic_add(&grad(node, equation, dir),
                        //  coeffs_transpose(index, dir) * d);
      }
    };
    int num_work_items = num_non_zeros * N;
    Kokkos::parallel_for("calc_grad_flat", num_work_items, calc_node_grad);
                        //  Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                        //      {0, 0}, {num_non_zeros, N}),
                        //  calc_node_grad);
  }

  template <size_t N>
  void calcMultipleGrads(Kokkos::View<double *[N]> fields,
                         const vul::Grid<vul::Device> &grid,
                         Kokkos::View<double *[N][3]> grad) {

    long num_nodes = grid.node_to_cell.num_rows;
    Kokkos::deep_copy(grad, 0.0);

    auto n2c            = grid.node_to_cell;
    auto calc_node_grad = KOKKOS_CLASS_LAMBDA(int n, int e, int dir) {
      auto start = n2c.rowStart(n);
      auto end   = n2c.rowEnd(n);
      for (int index = start; index < end; index++) {
        long neighbor = n2c.cols(index);
        double d      = fields(neighbor, e);
        grad(n, e, dir) += coeffs(index, dir) * d;
      }
    };
    Kokkos::parallel_for(
        "calc_grad_node",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {num_nodes, N, 3}),
        calc_node_grad);
  }

  template <typename GetFieldValue>
  void calcGrad(GetFieldValue get_field_value,
                const vul::Grid<vul::Device> &grid,
                Kokkos::View<double *[3]> grad) const {
    long num_nodes = grid.node_to_cell.num_rows;
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

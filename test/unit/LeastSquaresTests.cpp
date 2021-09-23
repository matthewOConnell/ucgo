#include <catch.hpp>
#include <vul/Decompositions.h>
#include <vul/DynamicMatrix.h>
#include <vul/Grid.h>
#include <vul/Macros.h>

namespace vul {
class LeastSquares {
public:
  LeastSquares(const vul::Grid<vul::Host> &grid)
      : coeffs("lsq_coeffs", grid.node_to_cell.num_non_zero) {
    auto coeffs_host =  Kokkos::View<double* [4], vul::Host::space>("lsq-host-coeffs", grid.node_to_cell.num_non_zero);
    auto cell_centroids  = grid.cell_centroids;
    auto getCellCentroid = KOKKOS_LAMBDA(int cell) {
      vul::Point<double> p;
      p.x = cell_centroids(cell, 0);
      p.y = cell_centroids(cell, 1);
      p.z = cell_centroids(cell, 2);
      return p;
    };

    auto points   = grid.points;
    auto getPoint = KOKKOS_LAMBDA(int node) {
      vul::Point<double> p;
      p.x = points(node, 0);
      p.y = points(node, 1);
      p.z = points(node, 2);
      return p;
    };

    for (int node = 0; node < grid.node_to_cell.num_rows; node++) {
      auto get_neighbor_weight = KOKKOS_LAMBDA(int i) { return 1.0; };
      setLSQWeights(getCellCentroid, grid.node_to_cell(node),
                    get_neighbor_weight, getPoint(node), coeffs_host, grid.node_to_cell.rowStart(node));
    }
    vul::force_copy(coeffs, coeffs_host);
  }

public:
  Kokkos::View<double *[4], vul::Device::space> coeffs;

  template <typename GetFieldValue>
  void calcGrad(GetFieldValue get_field_value,
                const vul::Grid<vul::Device> &grid,
                Kokkos::View<double *[3]> grad) const {
    long num_stencils = grid.node_to_cell.num_rows;
    for (int c = 0; c < num_stencils; c++) {
      grad(c, 0) = 0.0;
      grad(c, 1) = 0.0;
      grad(c, 2) = 0.0;
    }

    auto calc_node_grad = KOKKOS_LAMBDA(int n) {
      auto row = grid.node_to_cell(n);
      for(int i = 0; i < row.size; i++){
        int index = row.row_index_start + i;
        long neighbor = row(i);
        double d = get_field_value(neighbor);
        grad(n, 0) += coeffs(index, 1) * d;
        grad(n, 1) += coeffs(index, 2) * d;
        grad(n, 2) += coeffs(index, 3) * d;
      }
    };
    Kokkos::parallel_for("calc grad", grid.node_to_cell.num_rows,
                         calc_node_grad);
  }

  template <typename getPoint, typename getWeight, typename Point, typename Row>
  void setLSQWeights(getPoint get_neighbor_point, Row row,
                     getWeight get_neighbor_weight, const Point &center_point, Kokkos::View<double*[4]> coeffs_write, long write_offset) {
    using Matrix = vul::DynamicMatrix<double>;
    Matrix A(row.size, 4);
    for (int i = 0; i < row.size; ++i) {
      auto neighbor = row(i);
      auto w        = get_neighbor_weight(neighbor);
//      auto distance = get_neighbor_point(neighbor) - center_point;
      auto p = get_neighbor_point(neighbor);
      A(i, 0)   = w * 1.0;
      A(i, 1)   = w * p.x;
      A(i, 2)   = w * p.y;
      A(i, 3)   = w * p.z;
    }
    Matrix Q, R;
    std::tie(Q, R) = vul::householderDecomposition(A);
    auto Ainv = vul::calcPseudoInverse(Q, R);
    VUL_ASSERT(Ainv.columns == row.size, "Ainv should be same size ("+std::to_string(Ainv.rows)+") as rows" + std::to_string(row.size) );
    for (int i = 0; i < row.size; ++i) {
      auto neighbor = row(i);
      auto w           = get_neighbor_weight(neighbor);
      coeffs_write(write_offset + i, 0) = w * Ainv(0, i);
      coeffs_write(write_offset + i, 1) = w * Ainv(1, i);
      coeffs_write(write_offset + i, 2) = w * Ainv(2, i);
      coeffs_write(write_offset + i, 3) = w * Ainv(3, i);
    }
  }
};
} // namespace vul

TEST_CASE("Least squares weight calculation") {
  auto grid_host = vul::Grid<vul::Host>(10, 10, 10);
  auto grid      = vul::Grid<vul::Device>(grid_host);
  vul::LeastSquares grad_calculator(grid);

  Kokkos::View<double *[3], vul::Device::space> grad("test gradient",
                                                     grid.numPoints());

  auto centroids       = grid.cell_centroids;
  auto getCellCentroid = KOKKOS_LAMBDA(int cell) {
    vul::Point<double> p;
    p.x = centroids(cell, 0);
    p.y = centroids(cell, 1);
    p.z = centroids(cell, 2);
    return p;
  };
  auto get_field_at_cell = KOKKOS_LAMBDA(int c) {
    auto p = getCellCentroid(c);
    return 3.8 * p.x + 4.5*p.y - 9.7*p.z;
  };
  grad_calculator.calcGrad(get_field_at_cell, grid, grad);
  auto grad_mirror = create_mirror(grad);
  vul::force_copy(grad_mirror, grad);
  for (int n = 0; n < grid.numPoints(); n++) {
    REQUIRE(grad_mirror(n, 0) == Approx(3.8));
    REQUIRE(grad_mirror(n, 1) == Approx(4.5));
    REQUIRE(grad_mirror(n, 2) == Approx(-9.7));
  }
}

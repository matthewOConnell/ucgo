#if 0
#include <catch.hpp>
#include <vul/Decompositions.h>
#include <vul/DynamicMatrix.h>
#include <vul/Grid.h>

namespace vul {
class LeastSquares {
public:
  LeastSquares(const vul::Grid &grid)
      : coeffs("lsq_coeffs", grid.node_to_cell.num_non_zero) {
    auto cell_centroids  = grid.cell_centroids.d_view;
    auto getCellCentroid = KOKKOS_LAMBDA(int cell) {
      vul::Point<double> p;
      p.x = cell_centroids(cell, 0);
      p.y = cell_centroids(cell, 1);
      p.z = cell_centroids(cell, 2);
      return p;
    };

    auto points  = grid.points.d_view;
    auto getPoint = KOKKOS_LAMBDA(int node){
      vul::Point<double> p;
      p.x = points(node, 0);
      p.y = points(node, 1);
      p.z = points(node, 2);
      return p;
    };

    auto n2c_rows = grid.node_to_cell.rows.d_view;
    auto n2c_cols = grid.node_to_cell.cols.d_view;
    for(int node = 0; node < grid.numPoints(); node++){
      int num_neighbors = grid.node_to_cell.rows.d_view(node+1) - grid.node_to_cell.rows.d_view(node);
      auto get_neighbor_weight = KOKKOS_LAMBDA(int i){
        long index = n2c_rows(node) + i;
        long neighbor = n2c_cols(index);
        // we could compute an inverse distance here.
        return 1.0;
      };
      setLSQWeights(getCellCentroid, num_neighbors, get_neighbor_weight, getPoint(node));
    }
  }

public:
  Kokkos::DualView<double *[4]> coeffs;

  template <typename GetFieldValue>
  void calcGrad(GetFieldValue get_field_value, const vul::Grid &grid,
                Kokkos::View<double *[3]> grad) const {
    long num_stencils = grid.node_to_cell.num_rows;
    for (int c = 0; c < num_stencils; c++) {
      grad(c, 0) = 0.0;
      grad(c, 1) = 0.0;
      grad(c, 2) = 0.0;
    }

    auto calc_node_grad = KOKKOS_LAMBDA(int n) {
      for (long i = grid.node_to_cell.rows.d_view(n);
           i < grid.node_to_cell.rows.d_view(n + 1); i++) {
        auto c   = grid.node_to_cell.cols.d_view(i);
        double d = get_field_value(c);
        grad(n, 0) += coeffs.d_view(i, 1) * d;
        grad(n, 1) += coeffs.d_view(i, 2) * d;
        grad(n, 2) += coeffs.d_view(i, 3) * d;
      }
    };
    Kokkos::parallel_for("calc grad", grid.node_to_cell.num_rows,
                         calc_node_grad);
  }

  template <typename getPoint, typename getWeight, typename Point>
  void setLSQWeights(getPoint get_neighbor_point, int number_of_neighbors,
                     getWeight get_neighbor_weight, const Point &center_point) {
    using Matrix = vul::DynamicMatrix<double>;
    Matrix A(number_of_neighbors, 4);
    for (int point = 0; point < number_of_neighbors; ++point) {
      auto w        = get_neighbor_weight(point);
      auto distance = get_neighbor_point(point) - center_point;
      A(point, 0)   = w * 1.0;
      A(point, 1)   = w * distance.x;
      A(point, 2)   = w * distance.y;
      A(point, 3)   = w * distance.z;
    }
    Matrix Q, R;
    std::tie(Q, R) = vul::householderDecomposition(A);

    auto Ainv = vul::calcPseudoInverse(Q, R);
    for (int point = 0; point < number_of_neighbors; ++point) {
      auto w                  = get_neighbor_weight(point);
      coeffs.h_view(point, 0) = w * Ainv(0, point);
      coeffs.h_view(point, 1) = w * Ainv(1, point);
      coeffs.h_view(point, 2) = w * Ainv(2, point);
      coeffs.h_view(point, 3) = w * Ainv(3, point);
    }

    Kokkos::deep_copy(coeffs.d_view, coeffs.h_view);
  }
};
} // namespace vul

TEST_CASE("Least squares weight calculation") {
  auto grid = vul::Grid(10, 10, 10);
  vul::LeastSquares grad_calculator(grid);

  Kokkos::DualView<double *[3]> grad("test gradient", grid.numPoints());

  auto centroids = grid.cell_centroids.h_view;
  auto getCellCentroid = KOKKOS_LAMBDA(int cell) {
    vul::Point<double> p;
    p.x = centroids(cell, 0);
    p.y = centroids(cell, 1);
    p.z = centroids(cell, 2);
    return p;
  };
  auto get_field_at_cell = KOKKOS_LAMBDA(int c) {
    auto p = getCellCentroid(c);
    return 3.8*p.x + 2.2*p.y -9.3*p.z;
  };
  grad_calculator.calcGrad(get_field_at_cell, grid, grad.d_view);
  Kokkos::deep_copy(grad.h_view, grad.d_view);
  for(int n = 0; n < grid.numPoints(); n++){
    printf("node %d, grad %lf %lf %lf\n", n, grad.h_view(n, 0), grad.h_view(n, 1), grad.h_view(n, 2));
  }
}
#endif
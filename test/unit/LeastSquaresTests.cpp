#include <catch.hpp>
#include <vul/Decompositions.h>
#include <vul/DynamicMatrix.h>
#include <vul/Grid.h>
#include <vul/Macros.h>
#include <vul/Gradients.h>

TEST_CASE("Least squares grad calculation using transpose"){
  auto grid_host = vul::Grid<vul::Host>(10, 10, 10);
  vul::Grid<vul::Device> grid_device;
  grid_device.deep_copy(grid_host);
  // auto grid      = vul::Grid<vul::Device>(grid_host);
  vul::LeastSquares grad_calculator(grid_host);

  auto centroids       = grid_host.cell_centroids;
  Kokkos::View<double *[1], vul::Device::space> linear_field("linear field", grid_host.numCells());

  auto set_function = KOKKOS_LAMBDA(int cell){
    vul::Point<double> p{centroids(cell, 0), centroids(cell, 1), centroids(cell, 2)};
    linear_field(cell, 0) = 3.8 * p.x + 4.5*p.y - 9.7*p.z;
  };
  Kokkos::parallel_for("set linear field",grid_host.numCells(), set_function);

  Kokkos::View<double *[1][3], vul::Device::space> grad("test gradient",
                                                     grid_host.numPoints());
  grad_calculator.calcMultipleGrads_transpose<1>(linear_field, grid_device, grad);
  // grad_calculator.calcMultipleGrads<1>(linear_field, grid_device, grad);
  auto grad_mirror = create_mirror(grad);
  vul::force_copy(grad_mirror, grad);
  for (int n = 0; n < grid_host.numPoints(); n++) {
    REQUIRE(grad_mirror(n, 0, 0) == Approx(3.8));
    REQUIRE(grad_mirror(n, 0, 1) == Approx(4.5));
    REQUIRE(grad_mirror(n, 0, 2) == Approx(-9.7));
  }

}

TEST_CASE("Least squares weight calculation") {
  auto grid_host = vul::Grid<vul::Host>(10, 10, 10);
  vul::Grid<vul::Device> grid_device;
  grid_device.deep_copy(grid_host);
  // auto grid      = vul::Grid<vul::Device>(grid_host);
  vul::LeastSquares grad_calculator(grid_host);

  Kokkos::View<double *[3], vul::Device::space> grad("test gradient",
                                                     grid_host.numPoints());

  auto centroids       = grid_host.cell_centroids;
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
  grad_calculator.calcGrad(get_field_at_cell, grid_device, grad);
  auto grad_mirror = create_mirror(grad);
  vul::force_copy(grad_mirror, grad);
  for (int n = 0; n < grid_host.numPoints(); n++) {
    REQUIRE(grad_mirror(n, 0) == Approx(3.8));
    REQUIRE(grad_mirror(n, 1) == Approx(4.5));
    REQUIRE(grad_mirror(n, 2) == Approx(-9.7));
  }
}

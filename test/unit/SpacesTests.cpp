#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <stdio.h>
#include <string>
#include <vector>

template<typename Space>
class Grid {
public:
  template <typename OtherSpace>
  Grid(const Grid<OtherSpace>& rhs){
    if(std::is_same<OtherSpace, Space>::value){
      Kokkos::deep_copy(points, rhs.points);
    } else {
      auto points_host_mirror = create_mirror_view(rhs.points);
      Kokkos::deep_copy(points_host_mirror, rhs.points); // from device to mirror
      Kokkos::deep_copy(points, points_host_mirror); // from mirror to host
    }
  }
  Grid(int num_points) : points("points", num_points){
    Kokkos::DualView<double* [3]> my_points;
    for(int n = 0; n < num_points; n++){
      points(n, 0) = n;
      points(n, 1) = n;
      points(n, 2) = n;
    }
  }

  Kokkos::View<double* [3], Space> points;

  KOKKOS_FUNCTION double mag(int n) const {
    double m = points(n, 0);
    return m;
  }
};

void go(){
  using Host = Kokkos::DefaultHostExecutionSpace::memory_space;
  using Device = Kokkos::DefaultExecutionSpace::memory_space;
  int num_points = 10;
  auto grid_1 = Grid<Host>(num_points);
  auto grid_2 = Grid<Device>(grid_1);

  printf("Type %s\n", typeid(decltype(grid_1.points)).name());

  double norm    = 0.0;
  auto calc_norm = KOKKOS_LAMBDA(int n, double &norm) {
      norm += grid_1.mag(n);
  };
  Kokkos::parallel_reduce("norm", num_points, calc_norm, norm);
  printf("norm = %e\n", norm);
}

int main(int num_args, const char *args[]) {

  Kokkos::InitArguments arguments;
  arguments.disable_warnings = true;
  Kokkos::initialize(arguments);

  std::ostringstream msg;
#if defined(__CUDACC__)
  Kokkos::Cuda::print_configuration(msg);
#endif
  std::cout << msg.str() << std::endl;

  go();

  Kokkos::finalize();
  return 0;
}

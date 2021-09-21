#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <stdio.h>
#include <string>
#include <vector>


template <typename View1, typename View2>
void for_real_copy(View1 to, View2 from){
    using WriteSpace = typename View1::memory_space;
    using ReadSpace = typename View2::memory_space;
    using HostSpace = Kokkos::DefaultHostExecutionSpace::memory_space;
    using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;
    // same to same
    if(std::is_same<ReadSpace, WriteSpace>::value){
      Kokkos::deep_copy(to, from);
      return;
    } 
    // host to device
    if(std::is_same<WriteSpace, DeviceSpace>::value){
      auto mirror = create_mirror_view(to);
      Kokkos::deep_copy(mirror, from); 
      Kokkos::deep_copy(to, mirror);
      return;
    }
    // device to host
    if(std::is_same<WriteSpace, HostSpace>::value){
      auto mirror = create_mirror_view(from);
      Kokkos::deep_copy(mirror, from); 
      Kokkos::deep_copy(to, mirror); 
      return;
    }
    printf("You should not be here.\n");
}

template<typename Space>
class Grid {
public:
  template <typename OtherSpace>
  Grid(const Grid<OtherSpace>& rhs){
      points = Kokkos::View<double*[3], Space>("points_gpu", rhs.points.extent(0));
      for_real_copy(points, rhs.points);
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

  double norm    = 0.0;
  auto calc_norm = KOKKOS_LAMBDA(int n, double &norm) {
      norm += grid_2.mag(n);
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

#include <catch.hpp>
#include <vul/Grid.h>

namespace vul {
class LeastSquares {
public:
  LeastSquares(const vul::Grid& grid){

  }
public:
  template <typename T> using Vec1D       = Kokkos::DualView<T *>;
  Vec1D<double> coeffs;
};
}

TEST_CASE("Least squares weight calculation"){
  auto grid = vul::Grid(10,10,10);

}
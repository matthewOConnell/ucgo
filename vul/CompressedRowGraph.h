#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <vector>
#include "Macros.h"

namespace vul {

template <typename Space>
class CompressedRowGraph {
public:
  // Template magic to determine if this object is on the Host or Device
  using space = typename Space::space;
  CompressedRowGraph() = default;

  template <typename OtherSpace>
  CompressedRowGraph(const CompressedRowGraph<OtherSpace>& g) {
    num_non_zero = g.num_non_zero;
    num_rows = g.num_rows;
    rows     = Vec1D<int>("crs_ia", num_rows + 1);
    cols     = Vec1D<int>("crs_ja", num_non_zero);
    vul::force_copy(rows, g.rows);
    vul::force_copy(cols, g.cols);
  }

  template <typename SubContainer>
  CompressedRowGraph(const std::vector<SubContainer> &graph) {
    num_non_zero = 0;
    for (auto &row : graph) {
      num_non_zero += row.size();
    }
    num_rows = graph.size();
    rows     = Vec1D<int>("crs_ia", num_rows + 1);
    cols     = Vec1D<int>("crs_ja", num_non_zero);

    int next_ja = 0;
    for (long i = 0; i < long(graph.size()); i++) {
      rows(i + 1) = rows(i) + long(graph[i].size());
      for (int id : graph[i]) {
        cols(next_ja++) = id;
      }
    }
  }

public:
  template <typename T> using Vec1D = Kokkos::View<T *, space>;
  long num_rows                     = 0;
  long num_non_zero                 = 0;
  Vec1D<int> rows;
  Vec1D<int> cols;
};

} // namespace vul

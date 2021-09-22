#include <catch.hpp>
#include <vul/CompressedRowGraphTests.h>
#include <vul/Macros.h>

bool contains(const std::vector<int>& vector, int query){
  for(const auto& v : vector)
    if(v == query)
      return true;
  return false;
}

TEST_CASE("Can build a ragged array") {
  std::vector<std::vector<int>> node_to_cell = {
      {0, 1, 2}, {0, 1, 3}, {0, 2}, {1, 3}};

  vul::CompressedRowGraph<vul::Host> n2c(node_to_cell);
  long num_rows = n2c.num_rows;
  REQUIRE(num_rows == node_to_cell.size());
  REQUIRE(n2c.cols.extent_int(0) == 10);
  REQUIRE(n2c.num_non_zero == 10);

  auto rows = n2c.rows;
  auto cols = n2c.cols;
  for (int row = 0; row < num_rows; row++) {
    for (auto i = rows(row); i < rows(row + 1); i++) {
      auto column = cols(i);
      REQUIRE(contains(node_to_cell[row], column));
    }
  }
}
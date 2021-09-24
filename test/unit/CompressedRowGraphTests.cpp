#include <catch.hpp>
#include <vul/CompressedRowGraph.h>
#include <vul/Macros.h>

bool contains(const std::vector<int>& vector, int query){
  for(const auto& v : vector)
    if(v == query)
      return true;
  return false;
}

TEST_CASE("Can build device graph from host graph"){
  std::vector<std::vector<int>> node_to_cell = {
      {0, 1, 2}, {0, 1, 3}, {0, 2}, {1, 3}};
  vul::CompressedRowGraph<vul::Host> n2c_host(node_to_cell);
  auto n2c_device = vul::CompressedRowGraph<vul::Device>(n2c_host);
  REQUIRE(n2c_device.num_rows == n2c_host.num_rows);
  REQUIRE(n2c_device.num_non_zero == n2c_host.num_non_zero);
  REQUIRE(n2c_device.rows.extent_int(0) == n2c_host.rows.extent_int(0));
  REQUIRE(n2c_device.cols.extent_int(0) == n2c_host.cols.extent_int(0));
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

TEST_CASE("Can use accessor methods in a kokkos parallel region on the device"){
  std::vector<std::vector<int>> node_to_cell = {
      {0, 1, 2}, {0, 1, 3}, {0, 2}, {1, 3}};

  vul::CompressedRowGraph<vul::Host> n2c(node_to_cell);
  REQUIRE(n2c.size() == 4);

  REQUIRE(n2c.rowStart(0) == 0);
  REQUIRE(n2c.rowEnd(0) == 3);

  REQUIRE(n2c.rowLength(0) == 3);
  REQUIRE(n2c.rowLength(1) == 3);
  REQUIRE(n2c.rowLength(2) == 2);
  REQUIRE(n2c.rowLength(3) == 2);

  for(int r = 0; r < n2c.num_rows; r++){
    auto row = n2c(r);
    for(int i = 0; i < row.size(); i++){
      auto neighbor = row(i);
      REQUIRE(contains(node_to_cell[r], neighbor));
    }
  }
}
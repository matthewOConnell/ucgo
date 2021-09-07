#include <doctest.h>
#include <string>
#include <vul/Grid.h>

TEST_CASE("Can read primal data from a ugrid") {
  std::string assets_dir = ASSETS_DIR;
  std::string filename   = assets_dir + "/ramp.lb8.ugrid";
  vul::Grid grid(filename);
  REQUIRE(grid.count(vul::Grid::TRI) == 3176);
  REQUIRE(grid.count(vul::Grid::QUAD) == 16480);
  REQUIRE(grid.count(vul::Grid::TET) == 0);
  REQUIRE(grid.count(vul::Grid::PYRAMID) == 0);
  REQUIRE(grid.count(vul::Grid::PRISM) == 3176);
  REQUIRE(grid.count(vul::Grid::HEX) == 16000);

  REQUIRE(grid.cellType(0) == vul::Grid::PRISM);
  REQUIRE(grid.cellType(3176) == vul::Grid::HEX);
}

TEST_CASE("Can get faces of each celll type"){
  std::string assets_dir = ASSETS_DIR;
  std::string filename   = assets_dir + "/13-node.lb8.ugrid";
  vul::Grid grid(filename);

  REQUIRE(grid.count(vul::Grid::TRI) == 6);
  REQUIRE(grid.count(vul::Grid::QUAD) == 8);
  REQUIRE(grid.count(vul::Grid::TET) == 1);
  REQUIRE(grid.count(vul::Grid::PYRAMID) == 1);
  REQUIRE(grid.count(vul::Grid::PRISM) == 1);
  REQUIRE(grid.count(vul::Grid::HEX) == 1);
}

TEST_CASE("Can build faces from a ugrid") {
  std::string assets_dir = ASSETS_DIR;
  std::string filename   = assets_dir + "/shock.lb8.ugrid";
  vul::Grid grid(filename);
  REQUIRE(grid.count(vul::Grid::TRI) == 0);
  REQUIRE(grid.count(vul::Grid::QUAD) == 402);
  REQUIRE(grid.count(vul::Grid::TET) == 0);
  REQUIRE(grid.count(vul::Grid::PYRAMID) == 0);
  REQUIRE(grid.count(vul::Grid::PRISM) == 0);
  REQUIRE(grid.count(vul::Grid::HEX) == 100);

//  REQUIRE(grid.count(vul::Grid::FACE) == 402);
}

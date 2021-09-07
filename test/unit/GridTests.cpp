#include <doctest.h>
#include <string>
#include <vul/Grid.h>

TEST_CASE("Can read primal data from a ugrid") {
  std::string assets_dir = ASSETS_DIR;
  std::string filename   = assets_dir + "/ramp.lb8.ugrid";
  vul::Grid grid(filename);
  REQUIRE(grid.count(vul::TRI) == 3176);
  REQUIRE(grid.count(vul::QUAD) == 16480);
  REQUIRE(grid.count(vul::TET) == 0);
  REQUIRE(grid.count(vul::PYRAMID) == 0);
  REQUIRE(grid.count(vul::PRISM) == 3176);
  REQUIRE(grid.count(vul::HEX) == 16000);

  REQUIRE(grid.cellType(0) == vul::PRISM);
  REQUIRE(grid.cellType(3176) == vul::HEX);
}

TEST_CASE("Can get faces of each celll type"){
  std::string assets_dir = ASSETS_DIR;
  std::string filename   = assets_dir + "/13-node.lb8.ugrid";
  vul::Grid grid(filename);

  REQUIRE(grid.count(vul::TRI) == 6);
  REQUIRE(grid.count(vul::QUAD) == 8);
  REQUIRE(grid.count(vul::TET) == 1);
  REQUIRE(grid.count(vul::PYRAMID) == 1);
  REQUIRE(grid.count(vul::PRISM) == 1);
  REQUIRE(grid.count(vul::HEX) == 1);
}

TEST_CASE("Can build faces from a ugrid") {
  std::string assets_dir = ASSETS_DIR;
  std::string filename   = assets_dir + "/shock.lb8.ugrid";
  vul::Grid grid(filename);
  REQUIRE(grid.count(vul::TRI) == 0);
  REQUIRE(grid.count(vul::QUAD) == 402);
  REQUIRE(grid.count(vul::TET) == 0);
  REQUIRE(grid.count(vul::PYRAMID) == 0);
  REQUIRE(grid.count(vul::PRISM) == 0);
  REQUIRE(grid.count(vul::HEX) == 100);

//  REQUIRE(grid.count(vul::Grid::FACE) == 402);
}

TEST_CASE("vul::Cell exists"){
  std::string assets_dir = ASSETS_DIR;
  std::string filename   = assets_dir + "/13-node.lb8.ugrid";
  vul::Grid grid(filename);

  vul::Cell cell = grid.cell(0);
  REQUIRE(grid.cell(0).type() == vul::TET);
  REQUIRE(grid.cell(1).type() == vul::PYRAMID);
  REQUIRE(grid.cell(2).type() == vul::PRISM);
  REQUIRE(grid.cell(3).type() == vul::HEX);
  REQUIRE(grid.cell(4).type() == vul::TRI);
  REQUIRE(grid.cell(10).type() == vul::QUAD);

  REQUIRE(grid.cell(0) .numFaces() == 4);
  REQUIRE(grid.cell(1) .numFaces() == 5);
  REQUIRE(grid.cell(2) .numFaces() == 5);
  REQUIRE(grid.cell(3) .numFaces() == 6);
  REQUIRE(grid.cell(4) .numFaces() == 1);
  REQUIRE(grid.cell(10).numFaces() == 1);
}
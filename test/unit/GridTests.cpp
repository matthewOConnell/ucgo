#include <VectorStringMaker.h>
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

TEST_CASE("Can get faces of each celll type") {
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

  REQUIRE(grid.count(vul::FACE) == 402);
}

TEST_CASE("vul::Cell exists") {
  std::string assets_dir = ASSETS_DIR;
  std::string filename   = assets_dir + "/13-node.lb8.ugrid";
  vul::Grid grid(filename);

  vul::Cell cell = grid.cell(0);
  auto tet       = grid.cell(0);
  auto pyr       = grid.cell(1);
  auto pri       = grid.cell(2);
  auto hex       = grid.cell(3);
  auto tri       = grid.cell(4);
  auto qua       = grid.cell(10);
  REQUIRE(tet.type() == vul::TET);
  REQUIRE(pyr.type() == vul::PYRAMID);
  REQUIRE(pri.type() == vul::PRISM);
  REQUIRE(hex.type() == vul::HEX);
  REQUIRE(tri.type() == vul::TRI);
  REQUIRE(qua.type() == vul::QUAD);

  REQUIRE(tet.numFaces() == 4);
  REQUIRE(pyr.numFaces() == 5);
  REQUIRE(pri.numFaces() == 5);
  REQUIRE(hex.numFaces() == 6);
  REQUIRE(tri.numFaces() == 1);
  REQUIRE(qua.numFaces() == 1);

  REQUIRE(tet.face(0) == std::vector<int>{1, 4, 12});
  REQUIRE(pyr.face(0) == std::vector<int>{1, 4, 5, 2});
  REQUIRE(pri.face(0) == std::vector<int>{1, 2, 11, 10});
  REQUIRE(hex.face(0) == std::vector<int>{0, 3, 2, 1});
  REQUIRE(tri.face(0) == std::vector<int>{4, 1, 8});
  REQUIRE(qua.face(0) == std::vector<int>{3, 2, 1, 0});
}
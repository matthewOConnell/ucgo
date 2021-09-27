#include <catch.hpp>
#include <string>
#include <vul/Grid.h>
#include <vul/Macros.h>

TEST_CASE("Can automatically generate a cartesian grid"){
  vul::Grid<vul::Host> grid(2,2,2);
  REQUIRE(grid.count(vul::TRI) == 0);
  REQUIRE(grid.count(vul::QUAD) == 24);
  REQUIRE(grid.count(vul::TET) == 0);
  REQUIRE(grid.count(vul::PYRAMID) == 0);
  REQUIRE(grid.count(vul::PRISM) == 0);
  REQUIRE(grid.count(vul::HEX) == 8);
  REQUIRE(grid.count(vul::FACE) == 24 + 12);
  for(int h = 0; h < grid.count(vul::HEX); h++){
    REQUIRE(grid.cell_volume(h) == 1.0 / 8.0);
  }
  bool bot_corner_exists = false;
  bool top_corner_exists = false;
  for(int n = 0; n < grid.numPoints(); n++){
    auto p = grid.getPoint(n);
    if(p.x == Approx(0.0) and p.y == Approx(0.0) and p.z == Approx(0.0))
      bot_corner_exists = true;
    if(p.x == Approx(1.0) and p.y == Approx(1.0) and p.z == Approx(1.0))
      top_corner_exists = true;
  }
  REQUIRE(bot_corner_exists);
  REQUIRE(top_corner_exists);
}

TEST_CASE("Grid can report first instance of a boundary cell"){
  int num_cells_x = 2;
  int num_cells_y = 3;
  int num_cells_z = 4;
  // -- generate a cartesian hex grid
  auto grid = vul::Grid<vul::Host>(num_cells_x, num_cells_y, num_cells_z);
  int num_hexs = grid.count(vul::HEX);
  int num_quads = grid.count(vul::QUAD);
  int cell_start = grid.boundaryCellsStart();
  int cell_end = grid.boundaryCellsEnd();
  REQUIRE(cell_start == num_hexs);
  REQUIRE(cell_end == num_hexs + num_quads);
}

TEST_CASE("Can read primal data from a ugrid") {
  std::string assets_dir = ASSETS_DIR;
  std::string filename   = assets_dir + "/ramp.lb8.ugrid";
  vul::Grid<vul::Host> grid(filename);
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
  vul::Grid<vul::Host> grid(filename);

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
  vul::Grid<vul::Host> grid(filename);
  REQUIRE(grid.count(vul::TRI) == 0);
  REQUIRE(grid.count(vul::QUAD) == 402);
  REQUIRE(grid.count(vul::TET) == 0);
  REQUIRE(grid.count(vul::PYRAMID) == 0);
  REQUIRE(grid.count(vul::PRISM) == 0);
  REQUIRE(grid.count(vul::HEX) == 100);

  REQUIRE(grid.count(vul::FACE) == 501);
  REQUIRE(grid.face_area(0, 2) == -0.001);
}

TEST_CASE("vul::Cell exists") {
  std::string assets_dir = ASSETS_DIR;
  std::string filename   = assets_dir + "/13-node.lb8.ugrid";
  vul::Grid<vul::Host> grid(filename);

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

TEST_CASE("Can compute grid metrics of single cartesian hex"){
  auto grid = vul::Grid<vul::Host>(1,1,1);
  REQUIRE(grid.getCellCentroid(0).x == Approx(0.5));
  REQUIRE(grid.getCellCentroid(0).y == Approx(0.5));
  REQUIRE(grid.getCellCentroid(0).z == Approx(0.5));
}

TEST_CASE("Can compute grid metrics") {
  std::string assets_dir = ASSETS_DIR;
  std::string filename   = assets_dir + "/13-node.lb8.ugrid";
  vul::Grid<vul::Host> grid(filename);

  std::vector<int> face_nodes = {0, 3, 2, 1};
  auto face_area              = grid.calcFaceArea(face_nodes);
  REQUIRE(face_area.x == 0);
  REQUIRE(face_area.y == -1.0);
  REQUIRE(face_area.z == 0);

  REQUIRE(grid.cell_volume(0) == 1.0 / 24.0); // tet
  REQUIRE(grid.cell_volume(1) == 1.0 / 6.0);  // pyramid
  REQUIRE(grid.cell_volume(2) == 0.1);        // prism
  REQUIRE(grid.cell_volume(3) == 1.0);        // hex

  REQUIRE(grid.cell_centroids.extent_int(0) == grid.numCells());
  REQUIRE(grid.face_centroids.extent_int(0) == grid.numFaces());

}

TEST_CASE("Can convert to inf ordering") {
  std::string assets_dir = ASSETS_DIR;
  std::string filename   = assets_dir + "/13-node.lb8.ugrid";
  vul::Grid<vul::Host> grid(filename);

  // -- Triangles come first in Inf ordering
  REQUIRE(grid.getVulCellIdFromInfId(0) == 4);
  REQUIRE(grid.getVulCellIdFromInfId(1) == 5);
  REQUIRE(grid.getVulCellIdFromInfId(2) == 6);
  REQUIRE(grid.getVulCellIdFromInfId(3) == 7);
  REQUIRE(grid.getVulCellIdFromInfId(4) == 8);
  REQUIRE(grid.getVulCellIdFromInfId(5) == 9);

  // -- Quads come next
  REQUIRE(grid.getVulCellIdFromInfId(6) == 10);
  REQUIRE(grid.getVulCellIdFromInfId(7) == 11);
  REQUIRE(grid.getVulCellIdFromInfId(8) == 12);
  REQUIRE(grid.getVulCellIdFromInfId(9) == 13);
  REQUIRE(grid.getVulCellIdFromInfId(10) == 14);
  REQUIRE(grid.getVulCellIdFromInfId(11) == 15);
  REQUIRE(grid.getVulCellIdFromInfId(12) == 16);
  REQUIRE(grid.getVulCellIdFromInfId(13) == 17);

  // -- Then volume cells, Tet first
  REQUIRE(grid.getVulCellIdFromInfId(14) == 0);
  REQUIRE(grid.getVulCellIdFromInfId(15) == 1);
  REQUIRE(grid.getVulCellIdFromInfId(16) == 2);
  REQUIRE(grid.getVulCellIdFromInfId(17) == 3);
}

TEST_CASE("Can construct device grid from host grid"){
  vul::Grid<vul::Host> grid(10, 10, 10);
  auto grid_device = vul::Grid<vul::Device>(grid);

  auto face_to_nodes = create_mirror(grid_device.face_to_nodes);
  vul::force_copy(face_to_nodes, grid_device.face_to_nodes);
  REQUIRE(grid.count(vul::CellType::FACE) == face_to_nodes.extent(0));

  REQUIRE(grid_device.cell_face_neighbors.size() == grid.numCells());
  auto face_centroids = vul::create_host_copy(grid_device.face_centroids);
  REQUIRE(face_centroids.extent(0) == grid.numFaces());

  vul::Point<double> lo = {200,200,200};
  vul::Point<double> hi = {-200,-200,-200};

  for(int f = 0; f < face_centroids.extent(0); f++){
    lo.x = std::min(lo.x, face_centroids(f, 0));
    lo.y = std::min(lo.y, face_centroids(f, 1));
    lo.z = std::min(lo.z, face_centroids(f, 2));

    hi.x = std::max(hi.x, face_centroids(f, 0));
    hi.y = std::max(hi.y, face_centroids(f, 1));
    hi.z = std::max(hi.z, face_centroids(f, 2));
  }

  REQUIRE(lo.x >= 0.0);
  REQUIRE(lo.y >= 0.0);
  REQUIRE(lo.z >= 0.0);

  REQUIRE(hi.x <= 1.0);
  REQUIRE(hi.y <= 1.0);
  REQUIRE(hi.z <= 1.0);
}

TEST_CASE("Can create cell to node as compressed row storage"){
  vul::Grid<vul::Host> grid(10, 10, 10);
  auto grid_device = vul::Grid<vul::Device>(grid);
}
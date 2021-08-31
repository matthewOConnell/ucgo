#include <doctest.h>
#include <string>
#include <vul/Grid.h>

TEST_CASE("First test"){
    std::string assets_dir = ASSETS_DIR;
    std::string filename = assets_dir + "/ramp.lb8.ugrid";
    vul::Grid grid(filename);
    REQUIRE(grid.count(vul::Grid::TRI) == 3176);
    REQUIRE(grid.count(vul::Grid::QUAD) == 16480);
    REQUIRE(grid.count(vul::Grid::TET) == 0);
    REQUIRE(grid.count(vul::Grid::PYRAMID) == 0);
    REQUIRE(grid.count(vul::Grid::PRISM) == 3176);
    REQUIRE(grid.count(vul::Grid::HEX) == 16000);

    grid.printSummary();
}


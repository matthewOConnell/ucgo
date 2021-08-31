#pragma once
#include <string>
#include <Kokkos_Core.hpp>

namespace vul {
class Grid {
  public:
    enum CellType {TRI, QUAD, TET, PYRAMID, PRISM, HEX};
    Grid(std::string filename);

    int count(CellType type) const;
  private:
    using Vec2D = Kokkos::View<double**>;
    Vec2D points;
    Vec2D tris;
    Vec2D quads;
    Vec2D tets;
    Vec2D pyramids;
    Vec2D prisms;
    Vec2D hexs;

};
}


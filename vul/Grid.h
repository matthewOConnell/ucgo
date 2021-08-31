#pragma once
#include <string>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

namespace vul {
class Grid {
  public:
    template <typename T>
    using Vec1D = Kokkos::DualView<T*>;
    template <typename T>
    using Vec2D = Kokkos::DualView<T**>;

    enum CellType {TRI, QUAD, TET, PYRAMID, PRISM, HEX};
    Grid(std::string filename);

    int count(CellType type) const;
    int typeLength(CellType type) const;

    Vec2D<int> getCellArray(CellType type);

    void print() const;
  private:
    Vec2D<double> points;
    Vec2D<int> tris;
    Vec2D<int> quads;
    Vec2D<int> tets;
    Vec2D<int> pyramids;
    Vec2D<int> prisms;
    Vec2D<int> hexs;
    Vec1D<int> tri_tags;
    Vec1D<int> quad_tags;

    void readPoints(FILE* fp);
    void readCells(FILE* fp);
    void readCells(FILE* fp, CellType type);
    void readTags(FILE* fp, CellType type);

};
}


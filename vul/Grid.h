#pragma once
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <string>

namespace vul {
class Grid {
public:
  template <typename T> using Vec1D      = Kokkos::DualView<T *>;
  template <typename T> using Vec2D      = Kokkos::DualView<T **>;
  template <typename T> using FaceVector = Kokkos::DualView<T *[2]>;

  enum CellType { TRI, QUAD, TET, PYRAMID, PRISM, HEX, FACE };
  Grid(std::string filename);

  int count(CellType type) const;
  int typeLength(CellType type) const;
  int cellLength(int cell_id) const;
  CellType cellType(int cell_id) const;

  Vec2D<int> getCellArray(CellType type);

  std::pair<CellType, int> cellIdToTypeAndIndexPair(int cell_id) const;

  void printSummary() const;

  int numTets() const;
  int numPyramids() const;
  int numPrisms() const;
  int numHexs() const;
  int numTris() const;
  int numQuads() const;


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
  FaceVector<int> face_to_cell;

  void readPoints(FILE *fp);
  void readCells(FILE *fp);
  void readCells(FILE *fp, CellType type);
  void readTags(FILE *fp, CellType type);
};
} // namespace vul

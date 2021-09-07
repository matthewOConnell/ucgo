#pragma clang diagnostic push
#pragma ide diagnostic ignored "modernize-use-nodiscard"
#pragma once
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <set>
#include <string>

namespace vul {
enum CellType { TRI, QUAD, TET, PYRAMID, PRISM, HEX, FACE };
class Cell {
public:
  Cell(CellType type, const std::vector<int> &nodes);
  Cell(CellType type, const int *nodes);
  CellType type() const { return _type; }
  int numNodes() const { return int(cell_nodes.size()); }
  int node(int n) const { return cell_nodes[n]; }

  int numFaces();

  std::vector<int> face(int i) const;

private:
  CellType _type;
  std::vector<int> cell_nodes;
};
class Grid {
public:
  template <typename T> using Vec1D      = Kokkos::DualView<T *>;
  template <typename T> using Vec2D      = Kokkos::DualView<T **>;
  template <typename T> using FaceVector = Kokkos::DualView<T *[2]>;

  Grid(std::string filename);

  int count(CellType type) const;
  static int typeLength(CellType type);
  int cellLength(int cell_id) const;
  CellType cellType(int cell_id) const;
  Cell cell(int cell_id) const;

  Vec2D<int> getCellArray(CellType type);

  std::pair<CellType, int> cellIdToTypeAndIndexPair(int cell_id) const;

  void printSummary() const;

  int numCells() const;
  int numVolumeCells() const;
  int numPoints() const;
  int numTets() const;
  int numPyramids() const;
  int numPrisms() const;
  int numHexs() const;
  int numTris() const;
  int numQuads() const;

  void getCell(int cell_id, int *cell_nodes) const;

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
  void buildFaces();

  std::vector<std::set<int>> buildNodeToCell();
};

} // namespace vul

#pragma clang diagnostic pop
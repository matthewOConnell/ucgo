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
  inline Cell(CellType type, const std::vector<int> &nodes)
      : _type(type), cell_nodes(nodes) {}
  inline Cell(CellType type, const int *nodes) : _type(type) {
    switch (type) {
    case TRI:
      cell_nodes = std::vector<int>{nodes[0], nodes[1], nodes[2]};
      return;
    case QUAD:
      cell_nodes = std::vector<int>{nodes[0], nodes[1], nodes[2], nodes[3]};
      return;
    case TET:
      cell_nodes = std::vector<int>{nodes[0], nodes[1], nodes[2], nodes[3]};
      return;
    case PYRAMID:
      cell_nodes =
          std::vector<int>{nodes[0], nodes[1], nodes[2], nodes[3], nodes[4]};
      return;
    case PRISM:
      cell_nodes = std::vector<int>{nodes[0], nodes[1], nodes[2],
                                    nodes[3], nodes[4], nodes[5]};
      return;
    case HEX:
      cell_nodes = std::vector<int>{nodes[0], nodes[1], nodes[2], nodes[3],
                                    nodes[4], nodes[5], nodes[6], nodes[7]};
      return;
    }
  }
  CellType type() const { return _type; }
  int numNodes() const { return int(cell_nodes.size()); }
  int node(int n) const { return cell_nodes[n]; }

  int numFaces() {
    switch (type()) {
    case TET: return 4;
    case PYRAMID: return 5;
    case PRISM: return 5;
    case HEX: return 6;
    case TRI: return 1;
    case QUAD: return 1;
    }
  }

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
#pragma once
#include "Point.h"
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
  template <typename T> using Vec1D       = Kokkos::DualView<T *>;
  template <typename T> using Vec2D       = Kokkos::DualView<T **>;
  template <typename T> using PointVector = Kokkos::DualView<T *[3]>;
  using FaceToCells                       = Kokkos::DualView<int *[2]>;
  using FaceArea                          = Kokkos::DualView<double *[3]>;

  Grid(std::string filename);

  int count(CellType type) const;
  static int typeLength(CellType type);
  int cellLength(int cell_id) const;
  CellType cellType(int cell_id) const;
  Cell cell(int cell_id) const;

  Vec2D<int> getCellArray(CellType type);

KOKKOS_FUNCTION std::pair<vul::CellType, int>
cellIdToTypeAndIndexPair(int cell_id) const {
  int orig_cell_id = cell_id;
  if (cell_id < numTets())
    return {TET, cell_id};
  cell_id -= numTets();
  if (cell_id < numPyramids())
    return {PYRAMID, cell_id};
  cell_id -= numPyramids();
  if (cell_id < numPrisms())
    return {PRISM, cell_id};
  cell_id -= numPrisms();
  if (cell_id < numHexs())
    return {HEX, cell_id};
  cell_id -= numHexs();
  if (cell_id < numTris())
    return {TRI, cell_id};
  cell_id -= numTris();
  if (cell_id < numQuads())
    return {QUAD, cell_id};
  cell_id -= numQuads();
  //VUL_ASSERT(false, "Could not find type of cell_id " + std::to_string(orig_cell_id));
}
  Point<double> getPoint(int node_id) const;

  void printSummary() const;

KOKKOS_FUNCTION int numPoints() const { return points.extent_int(0); }
KOKKOS_FUNCTION int numTets() const { return tets.extent_int(0); }
KOKKOS_FUNCTION int numPyramids() const { return pyramids.extent_int(0); }
KOKKOS_FUNCTION int numPrisms() const { return prisms.extent_int(0); }
KOKKOS_FUNCTION int numHexs() const { return hexs.extent_int(0); }
KOKKOS_FUNCTION int numTris() const { return tris.extent_int(0); }
KOKKOS_FUNCTION int numQuads() const { return quads.extent_int(0); }
KOKKOS_FUNCTION int numCells() const {
  return numTets() + numPyramids() + numPrisms() + numHexs() + numTris() +
         numQuads();
}
KOKKOS_FUNCTION int numVolumeCells() const {
  return numTets() + numPyramids() + numPrisms() + numHexs();
}

  void getCell(int cell_id, int *cell_nodes) const;
  void getCell(int cell_id, std::vector<int> &cell_nodes) const;
  int getVulCellIdFromInfId(int inf_id) const;

public:
  PointVector<double> points;
  FaceArea face_area;
  Vec2D<int> tris;
  Vec2D<int> quads;
  Vec2D<int> tets;
  Vec2D<int> pyramids;
  Vec2D<int> prisms;
  Vec2D<int> hexs;
  Vec1D<int> tri_tags;
  Vec1D<int> quad_tags;
  Vec1D<double> cell_volume;
  FaceToCells face_to_cell;

  std::vector<std::vector<int>> cell_face_neighbors;
  std::vector<std::set<int>> node_to_cell;

  void readPoints(FILE *fp);
  void readCells(FILE *fp);
  void readCells(FILE *fp, CellType type);
  void readTags(FILE *fp, CellType type);
  void buildFaces();
  std::vector<std::vector<int>> buildFaceNeighbors();

  std::vector<std::set<int>> buildNodeToCell();
  std::vector<int> getNodeNeighborsOfCell(const std::vector<int> &cell_nodes,
                                          int cell_id);
  std::vector<int> getFaceNeighbors(CellType type,
                                    const std::vector<int> &cell_nodes,
                                    const std::vector<int> &candidates);
  int findFaceNeighbor(const std::vector<int> &candidates,
                       const std::vector<int> &face_nodes);
  bool cellContainsFace(const std::vector<int> &neighbor_nodes,
                        const std::vector<int> &face_nodes);
  Point<double> calcFaceArea(const std::vector<int> &face_nodes) const;
  void computeCellVolumes();
  double computeTetVolume(const Point<double> &a, const Point<double> &b,
                          const Point<double> &c, const Point<double> &d);
  double computeTetVolume(int t);
  double computePyramidVolume(int p);
  double computePrismVolume(int p);
  double computeHexVolume(int p);
};

} // namespace vul

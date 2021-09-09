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

  std::pair<CellType, int> cellIdToTypeAndIndexPair(int cell_id) const;
  Point<double> getPoint(int node_id) const;

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

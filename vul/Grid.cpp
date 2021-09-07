#include "Grid.h"
#include <set>

#include "Macros.h"

vul::Grid::Grid(std::string filename) {
  FILE *fp = fopen(filename.c_str(), "r");
  VUL_ASSERT(fp != nullptr, "Could not open file: " + filename);

  int nnodes, ntri, nquad, ntet, npyramid, nprism, nhex;
  fread(&nnodes, sizeof(int), 1, fp);
  fread(&ntri, sizeof(int), 1, fp);
  fread(&nquad, sizeof(int), 1, fp);
  fread(&ntet, sizeof(int), 1, fp);
  fread(&npyramid, sizeof(int), 1, fp);
  fread(&nprism, sizeof(int), 1, fp);
  fread(&nhex, sizeof(int), 1, fp);

  points   = Vec2D<double>("points", nnodes, 3);
  tris     = Vec2D<int>("tris", ntri, 3);
  quads    = Vec2D<int>("quads", nquad, 4);
  pyramids = Vec2D<int>("pyramids", npyramid, 5);
  prisms   = Vec2D<int>("prisms", nprism, 6);
  hexs     = Vec2D<int>("hexs", nhex, 8);

  tri_tags  = Vec1D<int>("tri_tags", ntri);
  quad_tags = Vec1D<int>("quad_tags", nquad);

  readPoints(fp);
  readCells(fp);
  fclose(fp);

  buildFaces();
}

void vul::Grid::readPoints(FILE *fp) {
  int nnodes = points.h_view.extent_int(0);
  std::vector<double> points_buffer(nnodes * 3);
  fread(points_buffer.data(), sizeof(double), 3 * nnodes, fp);
  for (int p = 0; p < nnodes; p++) {
    points.h_view(p, 0) = points_buffer[3 * p + 0];
    points.h_view(p, 1) = points_buffer[3 * p + 1];
    points.h_view(p, 2) = points_buffer[3 * p + 2];
  }
  Kokkos::deep_copy(points.d_view, points.h_view);
}
void vul::Grid::readTags(FILE *fp, CellType type) {
  Vec1D<int> tags;
  if (type == TRI) {
    tags = tri_tags;
  } else if (type == QUAD) {
    tags = quad_tags;
  }
  int ncells = tags.extent_int(0);
  std::vector<int> buffer(ncells);
  fread(buffer.data(), buffer.size(), sizeof(int), fp);
  for (int c = 0; c < ncells; c++) {
    tags.h_view(c) = buffer[c];
  }
  Kokkos::deep_copy(tags.d_view, tags.h_view);
}

void vul::Grid::readCells(FILE *fp, CellType type) {
  auto cells = getCellArray(type);
  int ncells = cells.extent_int(0);
  int length = typeLength(type);
  std::vector<int> buffer(ncells * length);
  fread(buffer.data(), buffer.size(), sizeof(int), fp);
  for (int c = 0; c < ncells; c++) {
    for (int i = 0; i < length; i++) {
      cells.h_view(c, i) = buffer[length * c + i];
    }
    if (type == PYRAMID) {
      //--- swap from AFLR to CGNS winding
      // all other cell types are the same
      std::swap(cells.h_view(c, 2), cells.h_view(c, 4));
      std::swap(cells.h_view(c, 1), cells.h_view(c, 3));
    }
  }
  Kokkos::deep_copy(cells.d_view, cells.h_view);
}

void vul::Grid::readCells(FILE *fp) {
  std::vector<int> buffer;

  readCells(fp, TRI);
  readCells(fp, QUAD);

  readTags(fp, TRI);
  readTags(fp, QUAD);

  readCells(fp, TET);
  readCells(fp, PYRAMID);
  readCells(fp, PRISM);
  readCells(fp, HEX);
}

int vul::Grid::count(vul::Grid::CellType type) const {
  switch (type) {
  case TRI: return tris.extent_int(0);
  case QUAD: return quads.extent_int(0);
  case TET: return tets.extent_int(0);
  case PYRAMID: return pyramids.extent_int(0);
  case PRISM: return prisms.extent_int(0);
  case HEX: return hexs.extent_int(0);
  case FACE: return face_to_cell.extent_int(0);
  default:
    throw std::logic_error("Unknown type requested in count" +
                           std::to_string(type));
  }
  return 0;
}
void vul::Grid::printSummary() const {
  printf("Num Node    %d\n", points.extent_int(0));
  printf("Num Tri     %d\n", tris.extent_int(0));
  printf("Num Quad    %d\n", quads.extent_int(0));
  printf("Num Pyramid %d\n", pyramids.extent_int(0));
  printf("Num Prism   %d\n", prisms.extent_int(0));
  printf("Num Hex     %d\n", hexs.extent_int(0));

  int first_few = std::min(3, points.extent_int(0));
  for (int p = 0; p < first_few; p++) {
    printf("point %d: %e %e %e\n", p, points.h_view(p, 0), points.h_view(p, 1),
           points.h_view(p, 2));
  }

  first_few = std::min(3, tris.extent_int(0));
  for (int p = 0; p < first_few; p++) {
    printf("tri %d: %d %d %d\n", p, tris.h_view(p, 0), tris.h_view(p, 1),
           tris.h_view(p, 2));
  }

  first_few = std::min(3, quads.extent_int(0));
  for (int p = 0; p < first_few; p++) {
    printf("quad %d: %d %d %d %d\n", p, quads.h_view(p, 0), quads.h_view(p, 1),
           quads.h_view(p, 2), quads.h_view(p, 3));
  }

  first_few = std::min(3, tets.extent_int(0));
  for (int p = 0; p < first_few; p++) {
    printf("tet %d: %d %d %d %d\n", p, tets.h_view(p, 0), tets.h_view(p, 1),
           tets.h_view(p, 2), tets.h_view(p, 3));
  }

  first_few = std::min(3, pyramids.extent_int(0));
  for (int p = 0; p < first_few; p++) {
    printf("pyramid %d: %d %d %d %d %d\n", p, pyramids.h_view(p, 0),
           pyramids.h_view(p, 1), pyramids.h_view(p, 2), pyramids.h_view(p, 3),
           pyramids.h_view(p, 4));
  }

  first_few = std::min(3, prisms.extent_int(0));
  for (int p = 0; p < first_few; p++) {
    printf("prism %d: %d %d %d %d %d %d\n", p, prisms.h_view(p, 0),
           prisms.h_view(p, 1), prisms.h_view(p, 2), prisms.h_view(p, 3),
           prisms.h_view(p, 4), prisms.h_view(p, 5));
  }
  first_few = std::min(3, hexs.extent_int(0));
  for (int p = 0; p < first_few; p++) {
    printf("hex %d: %d %d %d %d %d %d %d %d\n", p, hexs.h_view(p, 0),
           hexs.h_view(p, 1), hexs.h_view(p, 2), hexs.h_view(p, 3),
           hexs.h_view(p, 4), hexs.h_view(p, 5), hexs.h_view(p, 6),
           hexs.h_view(p, 7));
  }
}

vul::Grid::Vec2D<int> vul::Grid::getCellArray(vul::Grid::CellType type) {
  switch (type) {
  case TRI: return tris;
  case QUAD: return quads;
  case TET: return tets;
  case PYRAMID: return pyramids;
  case PRISM: return prisms;
  case HEX: return hexs;
  default:
    throw std::logic_error("Unknown type requested in count" +
                           std::to_string(type));
  }
}

int vul::Grid::typeLength(vul::Grid::CellType type) {
  switch (type) {
  case TRI: return 3;
  case QUAD: return 4;
  case TET: return 4;
  case PYRAMID: return 5;
  case PRISM: return 6;
  case HEX: return 8;
  default:
    throw std::logic_error("Unknown type requested in typeLength" +
                           std::to_string(type));
  }
}
int vul::Grid::cellLength(int cell_id) const {
  return typeLength(cellType(cell_id));
}
vul::Grid::CellType vul::Grid::cellType(int cell_id) const {
  auto pair = cellIdToTypeAndIndexPair(cell_id);
  return pair.first;
}
std::pair<vul::Grid::CellType, int>
vul::Grid::cellIdToTypeAndIndexPair(int cell_id) const {
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
  VUL_ASSERT(false, "Could not find type of cell_id" + std::to_string(cell_id));
}
int vul::Grid::numPoints() const { return points.extent_int(0); }
int vul::Grid::numTets() const { return tets.extent_int(0); }
int vul::Grid::numPyramids() const { return pyramids.extent_int(0); }
int vul::Grid::numPrisms() const { return prisms.extent_int(0); }
int vul::Grid::numHexs() const { return hexs.extent_int(0); }
int vul::Grid::numTris() const { return tris.extent_int(0); }
int vul::Grid::numQuads() const { return quads.extent_int(0); }

void vul::Grid::buildFaces() {
  auto n2c = buildNodeToCell();
}
void vul::Grid::getCell(int cell_id, int *cell_nodes) const {
  auto [cell_type, cell_index] = cellIdToTypeAndIndexPair(cell_id);
  switch (cell_type) {
  case TET: {
    for (int i = 0; i < 4; i++) {
      cell_nodes[i] = tets.h_view(cell_index, i);
    }
    return;
  }
  case PYRAMID: {
    for (int i = 0; i < 5; i++) {
      cell_nodes[i] = pyramids.h_view(cell_index, i);
    }
    return;
  }
  case PRISM: {
    for (int i = 0; i < 6; i++) {
      cell_nodes[i] = prisms.h_view(cell_index, i);
    }
    return;
  }
  case HEX: {
    for (int i = 0; i < 8; i++) {
      cell_nodes[i] = hexs.h_view(cell_index, i);
    }
    return;
  }
  case TRI: {
    for (int i = 0; i < 3; i++) {
      cell_nodes[i] = tris.h_view(cell_index, i);
    }
    return;
  }
  case QUAD: {
    for (int i = 0; i < 4; i++) {
      cell_nodes[i] = quads.h_view(cell_index, i);
    }
    return;
  }
  }
  VUL_ASSERT(false, "Could not match cell type: " + std::to_string(cell_type));
}
int vul::Grid::numCells() const {
  return numTets() + numPyramids() + numPrisms() + numHexs() + numTris() +
         numQuads();
}
int vul::Grid::numVolumeCells() const {
  return numTets() + numPyramids() + numPrisms() + numHexs();
}
std::vector<std::set<int>> vul::Grid::buildNodeToCell() {
  std::vector<std::set<int>> n2c(numPoints());
  std::vector<int> cell_nodes;
  cell_nodes.reserve(8);
  for (int c = 0; c < numCells(); c++) {
    getCell(c, cell_nodes.data());
    for (auto n : cell_nodes) {
      n2c[n].insert(c);
    }
  }
  return n2c;
}

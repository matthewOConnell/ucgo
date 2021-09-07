#include "Grid.h"
#include "Point.h"
#include <set>

#include "Macros.h"

template <typename T>
void getTetFace(const std::vector<T> &cell, int face_id, std::vector<T> &face) {
  face.resize(3);
  switch (face_id) {
  case 0:
    face[0] = cell[0];
    face[1] = cell[2];
    face[2] = cell[1];
    break;
  case 1:
    face[0] = cell[0];
    face[1] = cell[1];
    face[2] = cell[3];
    break;
  case 2:
    face[0] = cell[1];
    face[1] = cell[2];
    face[2] = cell[3];
    break;
  case 3:
    face[0] = cell[2];
    face[1] = cell[0];
    face[2] = cell[3];
    break;
  default: VUL_ASSERT(false, "unexpected tet face: " + std::to_string(face_id));
  }
}

template <typename T>
void getPyramidFace(const std::vector<T> &cell, int face_id,
                    std::vector<T> &face) {
  switch (face_id) {
  case 0:
    face.resize(4);
    face[0] = cell[0];
    face[1] = cell[3];
    face[2] = cell[2];
    face[3] = cell[1];
    break;
  case 1:
    face.resize(3);
    face[0] = cell[0];
    face[1] = cell[1];
    face[2] = cell[4];
    break;
  case 2:
    face.resize(3);
    face[0] = cell[1];
    face[1] = cell[2];
    face[2] = cell[4];
    break;
  case 3:
    face.resize(3);
    face[0] = cell[2];
    face[1] = cell[3];
    face[2] = cell[4];
    break;
  case 4:
    face.resize(3);
    face[0] = cell[3];
    face[1] = cell[0];
    face[2] = cell[4];
    break;
  default:
    VUL_ASSERT(false, "unexpected pyramid face: " + std::to_string(face_id));
  }
}

template <typename T>
void getPrismFace(const std::vector<T> &cell, int face_id,
                  std::vector<T> &face) {
  switch (face_id) {
  case 0:
    face.resize(4);
    face[0] = cell[0];
    face[1] = cell[1];
    face[2] = cell[4];
    face[3] = cell[3];
    break;
  case 1:
    face.resize(4);
    face[0] = cell[1];
    face[1] = cell[2];
    face[2] = cell[5];
    face[3] = cell[4];
    break;
  case 2:
    face.resize(4);
    face[0] = cell[2];
    face[1] = cell[0];
    face[2] = cell[3];
    face[3] = cell[5];
    break;
  case 3:
    face.resize(3);
    face[0] = cell[0];
    face[1] = cell[2];
    face[2] = cell[1];
    break;
  case 4:
    face.resize(3);
    face[0] = cell[3];
    face[1] = cell[4];
    face[2] = cell[5];
    break;
  default:
    VUL_ASSERT(false, "unexpected prism face: " + std::to_string(face_id));
  }
}

template <typename T>
void getHexFace(const std::vector<T> &cell, int face_id, std::vector<T> &face) {
  face.resize(4);
  switch (face_id) {
  case 0:
    face[0] = cell[0];
    face[1] = cell[3];
    face[2] = cell[2];
    face[3] = cell[1];
    break;
  case 1:
    face[0] = cell[0];
    face[1] = cell[1];
    face[2] = cell[5];
    face[3] = cell[4];
    break;
  case 2:
    face[0] = cell[1];
    face[1] = cell[2];
    face[2] = cell[6];
    face[3] = cell[5];
    break;
  case 3:
    face[0] = cell[2];
    face[1] = cell[3];
    face[2] = cell[7];
    face[3] = cell[6];
    break;
  case 4:
    face[0] = cell[0];
    face[1] = cell[4];
    face[2] = cell[7];
    face[3] = cell[3];
    break;
  case 5:
    face[0] = cell[4];
    face[1] = cell[5];
    face[2] = cell[6];
    face[3] = cell[7];
    break;
  default: VUL_ASSERT(false, "unexpected hex face: " + std::to_string(face_id));
  }
}

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

  points   = PointVector<double>("points", nnodes);
  tris     = Vec2D<int>("tris", ntri, 3);
  quads    = Vec2D<int>("quads", nquad, 4);
  tets     = Vec2D<int>("tets", ntet, 4);
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
      cells.h_view(c, i) = buffer[length * c + i] - 1;
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

int vul::Grid::count(vul::CellType type) const {
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
  first_few = std::min(3, int(cell_face_neighbors.size()));
  for(int c = 0; c < first_few; c++){
    printf("cell %d face neighbors: ", c);
    for(auto n : cell_face_neighbors[c]){
      printf("%d ", n);
    }
    printf("\n");
  }

  first_few = std::min(3, face_to_cell.extent_int(0));
  for(int f = 0; f < first_few; f++){
    printf("face %d %d\n", face_to_cell.h_view(f, 0), face_to_cell.h_view(f, 1));
  }
}

vul::Grid::Vec2D<int> vul::Grid::getCellArray(vul::CellType type) {
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

int vul::Grid::typeLength(vul::CellType type) {
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
vul::CellType vul::Grid::cellType(int cell_id) const {
  auto pair = cellIdToTypeAndIndexPair(cell_id);
  return pair.first;
}
std::pair<vul::CellType, int>
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

std::vector<std::vector<int>> vul::Grid::buildFaceNeighbors() {
  std::vector<std::vector<int>> neighbors(numCells());
  std::vector<int> cell;
  for (int cell_id = 0; cell_id < numCells(); cell_id++) {
    auto type = cellType(cell_id);
    cell.resize(cellLength(cell_id));
    getCell(cell_id, cell.data());
    auto candidates    = getNodeNeighborsOfCell(cell, cell_id);
    neighbors[cell_id] = getFaceNeighbors(type, cell, candidates);
  }
  return neighbors;
}

void vul::Grid::buildFaces() {
  node_to_cell        = buildNodeToCell();
  cell_face_neighbors = buildFaceNeighbors();

  // This math for num faces doesn't work in parallel if there are volume
  // cells that don't have face neighbors on rank.
  int num_faces = 6*numHexs() + 5*numPyramids() + 5*numPrisms() + 4*numTets() + numTris() + numQuads();
  num_faces /= 2;
  face_to_cell = FaceToCells("face_to_cell", num_faces);
  face_area = FaceArea("face_area", num_faces);

  int next_face = 0;
  for(int c = 0; c < numCells(); c++){
    for(int neighbor : cell_face_neighbors[c]){
      if(c < neighbor) {
        face_to_cell.h_view(next_face,0) = c;
        face_to_cell.h_view(next_face,1) = neighbor;
        Point<double> area;
        face_area.h_view(next_face, 0) = area.x;
        face_area.h_view(next_face, 1) = area.y;
        face_area.h_view(next_face, 2) = area.z;
        next_face++;
      }
    }
  }
  Kokkos::deep_copy(face_to_cell.d_view, face_to_cell.h_view);
  Kokkos::deep_copy(face_area.d_view, face_area.h_view);
}
void vul::Grid::getCell(int cell_id, std::vector<int>& cell_nodes) const {
  cell_nodes.resize(cellLength(cell_id));
  getCell(cell_id, cell_nodes.data());
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
    getCell(c, cell_nodes);
    for (auto n : cell_nodes) {
      n2c[n].insert(c);
    }
  }
  return n2c;
}
vul::Cell vul::Grid::cell(int cell_id) const {
  auto [type, index] = cellIdToTypeAndIndexPair(cell_id);
  auto length        = typeLength(type);
  std::vector<int> cell_nodes(length);
  getCell(cell_id, cell_nodes.data());
  return vul::Cell(type, cell_nodes);
}
template <class Container, class T>
inline bool isIn(const Container &container, T t) {
  for (auto u : container) {
    if (t == u)
      return true;
  }
  return false;
}
std::vector<int>
vul::Grid::getNodeNeighborsOfCell(const std::vector<int> &cell_nodes,
                                  int cell_id) {
  // This is convoluted for better performance, The original implementation used
  // std::set, which was ~4x slower.
  int neighbor_count = 0;
  for (int node : cell_nodes) {
    neighbor_count += node_to_cell.at(node).size();
  }

  std::vector<int> neighbors;
  neighbors.reserve(neighbor_count);
  for (int node : cell_nodes) {
    for (int c : node_to_cell[node]) {
      if (c != cell_id and not isIn(neighbors, c)) {
        neighbors.push_back(c);
      }
    }
  }
  neighbors.shrink_to_fit();
  return neighbors;
}
std::vector<int>
vul::Grid::getFaceNeighbors(vul::CellType type,
                            const std::vector<int> &cell_nodes,
                            const std::vector<int> &candidates) {

  Cell cell(type, cell_nodes);
  std::vector<int> cell_neighbors;
  for (int f = 0; f < cell.numFaces(); f++) {
    auto face         = cell.face(f);
    int face_neighbor = findFaceNeighbor(candidates, face);
    cell_neighbors.push_back(face_neighbor);
  }
  return cell_neighbors;
}
int vul::Grid::findFaceNeighbor(const std::vector<int> &candidates,
                                const std::vector<int> &face_nodes) {
  std::vector<int> neighbor;

  for (auto neighbor_id : candidates) {
    neighbor.resize(cellLength(neighbor_id));
    getCell(neighbor_id, neighbor.data());
    if (cellContainsFace(neighbor, face_nodes))
      return neighbor_id;
  }
  return -1;
}
bool vul::Grid::cellContainsFace(const std::vector<int> &cell,
                                 const std::vector<int> &face_nodes) {
  for (auto &id : face_nodes)
    if (std::find(cell.begin(), cell.end(), id) == cell.end())
      return false;
  return true;
}
vul::Point<double>
vul::Grid::calcFaceArea(const std::vector<int> &face_nodes) const {
  std::array<Point<double>, 4> face_points;
  bool is_quad = face_nodes.size() == 4;
  for(int i = 0; i < 4; i++){
    if(not is_quad) break; // break out for triangles;
    int n = face_nodes[i];
    Point<double> p;
    p.x = points.h_view(n, 0);
    p.y = points.h_view(n, 1);
    p.z = points.h_view(n, 2);
    face_points[i] = p;
  }

  Point<double> u = face_points[1] - face_points[0];
  Point<double> v = face_points[2] - face_points[0];

  Point<double> area = u.cross(v) * 0.5;

  if(is_quad){
    u = face_points[0] - face_points[3];
    v = face_points[2] - face_points[3];
    area = area + u.cross(v)*0.5;
  }

  return area;

}
std::vector<int> vul::Cell::face(int i) const {
  std::vector<int> face;
  switch (type()) {
  case TET: getTetFace(cell_nodes, i, face); return face;
  case PYRAMID: getPyramidFace(cell_nodes, i, face); return face;
  case PRISM: getPrismFace(cell_nodes, i, face); return face;
  case HEX: getHexFace(cell_nodes, i, face); return face;
  case TRI: return cell_nodes;
  case QUAD: return cell_nodes;
  default:
    VUL_ASSERT(false,
               "Could not find face for cell type " + std::to_string(type()));
  }
}
int vul::Cell::numFaces() {
  switch (type()) {
  case TET: return 4;
  case PYRAMID: return 5;
  case PRISM: return 5;
  case HEX: return 6;
  case TRI: return 1;
  case QUAD: return 1;
  default:
    VUL_ASSERT(false, "Num faces not known for unknown type: " +
                          std::to_string(type()));
  }
}
vul::Cell::Cell(vul::CellType type, const int *nodes) : _type(type) {
  switch (type) {
  case TRI: cell_nodes = std::vector<int>{nodes[0], nodes[1], nodes[2]}; return;
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
vul::Cell::Cell(vul::CellType type, const std::vector<int> &nodes)
    : _type(type), cell_nodes(nodes) {}

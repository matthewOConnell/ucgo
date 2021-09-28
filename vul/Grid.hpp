#include "CartBlock.h"
#include "Grid.h"
#include "Point.h"
#include <set>

#include "Macros.h"

template <typename Space>
template <typename OtherSpace>
void vul::Grid<Space>::deep_copy(const Grid<OtherSpace> &g) {
  points = PointVector<double>("points", g.points.extent(0));
  vul::force_copy(points, g.points);
  face_area = FaceArea("face_area", g.face_area.extent(0));
  vul::force_copy(face_area, g.face_area);
  tris = Vec2D<int>("tris", g.tris.extent(0), 3);
  vul::force_copy(tris, g.tris);
  quads = Vec2D<int>("quads", g.quads.extent(0), 4);
  vul::force_copy(quads, g.quads);
  tets = Vec2D<int>("tets", g.tets.extent(0), 4);
  vul::force_copy(tets, g.tets);
  pyramids = Vec2D<int>("pyramids", g.pyramids.extent(0), 5);
  vul::force_copy(pyramids, g.pyramids);
  prisms = Vec2D<int>("prisms", g.prisms.extent(0), 6);
  vul::force_copy(prisms, g.prisms);
  hexs = Vec2D<int>("hexs", g.hexs.extent(0), 8);
  vul::force_copy(hexs, g.hexs);
  tri_tags = Vec1D<int>("tri_tags", g.tri_tags.extent(0));
  vul::force_copy(tri_tags, g.tri_tags);
  quad_tags = Vec1D<int>("quad_tags", g.quad_tags.extent(0));
  vul::force_copy(quad_tags, g.quad_tags);
  cell_volume = Vec1D<double>("cell_volume", g.cell_volume.extent(0));
  vul::force_copy(cell_volume, g.cell_volume);
  face_to_cell = FaceToCells("face_to_cells", g.face_to_cell.extent(0));
  vul::force_copy(face_to_cell, g.face_to_cell);
  cell_centroids =
      PointVector<double>("cell_centroids", g.cell_centroids.extent(0));
  vul::force_copy(cell_centroids, g.cell_centroids);

  face_centroids =
      PointVector<double>("face_centroids", g.face_centroids.extent(0));
  vul::force_copy(face_centroids, g.face_centroids);

  face_to_nodes = FaceToNodes("face_to_nodes", g.face_to_nodes.extent(0));
  vul::force_copy(face_to_nodes, g.face_to_nodes);

  node_to_cell        = CompressedRowGraph<Space>(g.node_to_cell);
  cell_face_neighbors = CompressedRowGraph<Space>(g.cell_face_neighbors);
  cell_to_node        = CompressedRowGraph<Space>(g.cell_to_node);
}

template <typename Space>
vul::Grid<Space>::Grid(int n_cells_x, int n_cells_y, int n_cells_z) {
  Kokkos::Profiling::pushRegion("Cartesian Grid Generation");
  int num_nodes_x = n_cells_x + 1;
  int num_nodes_y = n_cells_y + 1;
  int num_nodes_z = n_cells_z + 1;

  int nnodes = num_nodes_x * num_nodes_y * num_nodes_z;
  int ntri   = 0;
  int nquad  = 2 * n_cells_x * n_cells_y + 2 * n_cells_x * n_cells_z +
              2 * n_cells_y * n_cells_z;
  int ntet     = 0;
  int npyramid = 0;
  int nprism   = 0;
  int nhex     = n_cells_x * n_cells_y * n_cells_z;
  points       = PointVector<double>("points", nnodes);
  tris         = Vec2D<int>("tris", ntri, 3);
  quads        = Vec2D<int>("quads", nquad, 4);
  tets         = Vec2D<int>("tets", ntet, 4);
  pyramids     = Vec2D<int>("pyramids", npyramid, 5);
  prisms       = Vec2D<int>("prisms", nprism, 6);
  hexs         = Vec2D<int>("hexs", nhex, 8);

  tri_tags  = Vec1D<int>("tri_tags", ntri);
  quad_tags = Vec1D<int>("quad_tags", nquad);

  setCartesianPoints(n_cells_x, n_cells_y, n_cells_z);
  setCartesianCells(n_cells_x, n_cells_y, n_cells_z);
  Kokkos::Profiling::popRegion();

  buildFaces();
  computeCellVolumes();
  computeCellCentroids();
}

template <typename Space> vul::Grid<Space>::Grid(std::string filename) {
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
  computeCellVolumes();
  computeCellCentroids();
}

template <typename Space> void vul::Grid<Space>::readPoints(FILE *fp) {
  if (not std::is_same<Space, vul::Host>::value) {
    VUL_ASSERT((std::is_same<Space, vul::Host>::value),
               "Cannot read points using a device grid");
  }
  int nnodes = points.extent_int(0);
  std::vector<double> points_buffer(nnodes * 3);
  fread(points_buffer.data(), sizeof(double), 3 * nnodes, fp);
  for (int p = 0; p < nnodes; p++) {
    points(p, 0) = points_buffer[3 * p + 0];
    points(p, 1) = points_buffer[3 * p + 1];
    points(p, 2) = points_buffer[3 * p + 2];
  }
}

template <typename Space>
void vul::Grid<Space>::readTags(FILE *fp, CellType type) {
  if (not std::is_same<Space, vul::Host>::value) {
    VUL_ASSERT((std::is_same<Space, vul::Host>::value),
               "Cannot readTags  using a device grid");
  }
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
    tags(c) = buffer[c];
  }
}

template <typename Space>
void vul::Grid<Space>::readCells(FILE *fp, CellType type) {
  if (not std::is_same<Space, vul::Host>::value) {
    VUL_ASSERT((std::is_same<Space, vul::Host>::value),
               "Cannot readCells  using a device grid");
  }
  auto cells = getCellArray(type);
  int ncells = cells.extent_int(0);
  int length = typeLength(type);
  std::vector<int> buffer(ncells * length);
  fread(buffer.data(), buffer.size(), sizeof(int), fp);
  for (int c = 0; c < ncells; c++) {
    for (int i = 0; i < length; i++) {
      cells(c, i) = buffer[length * c + i] - 1;
    }
    if (type == PYRAMID) {
      //--- swap from AFLR to CGNS winding
      // all other cell types are the same
      std::swap(cells(c, 2), cells(c, 4));
      std::swap(cells(c, 1), cells(c, 3));
    }
  }
}

template <typename Space> void vul::Grid<Space>::readCells(FILE *fp) {
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

template <typename Space>
int vul::Grid<Space>::count(vul::CellType type) const {
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

template <typename Space> void vul::Grid<Space>::printSummary() const {
  if (not std::is_same<Space, vul::Host>::value) {
    VUL_ASSERT((std::is_same<Space, vul::Host>::value),
               "Cannot printSummary using a device grid");
  }
  printf("Num Node    %d\n", points.extent_int(0));
  printf("Num Tri     %d\n", tris.extent_int(0));
  printf("Num Quad    %d\n", quads.extent_int(0));
  printf("Num Pyramid %d\n", pyramids.extent_int(0));
  printf("Num Prism   %d\n", prisms.extent_int(0));
  printf("Num Hex     %d\n", hexs.extent_int(0));

  int first_few = std::min(3, points.extent_int(0));
  for (int p = 0; p < first_few; p++) {
    printf("point %d: %e %e %e\n", p, points(p, 0), points(p, 1), points(p, 2));
  }

  first_few = std::min(3, tris.extent_int(0));
  for (int p = 0; p < first_few; p++) {
    printf("tri %d: %d %d %d\n", p, tris(p, 0), tris(p, 1), tris(p, 2));
  }

  first_few = std::min(3, quads.extent_int(0));
  for (int p = 0; p < first_few; p++) {
    printf("quad %d: %d %d %d %d\n", p, quads(p, 0), quads(p, 1), quads(p, 2),
           quads(p, 3));
  }

  first_few = std::min(3, tets.extent_int(0));
  for (int p = 0; p < first_few; p++) {
    printf("tet %d: %d %d %d %d\n", p, tets(p, 0), tets(p, 1), tets(p, 2),
           tets(p, 3));
  }

  first_few = std::min(3, pyramids.extent_int(0));
  for (int p = 0; p < first_few; p++) {
    printf("pyramid %d: %d %d %d %d %d\n", p, pyramids(p, 0), pyramids(p, 1),
           pyramids(p, 2), pyramids(p, 3), pyramids(p, 4));
  }

  first_few = std::min(3, prisms.extent_int(0));
  for (int p = 0; p < first_few; p++) {
    printf("prism %d: %d %d %d %d %d %d\n", p, prisms(p, 0), prisms(p, 1),
           prisms(p, 2), prisms(p, 3), prisms(p, 4), prisms(p, 5));
  }
  first_few = std::min(3, hexs.extent_int(0));
  for (int p = 0; p < first_few; p++) {
    printf("hex %d: %d %d %d %d %d %d %d %d\n", p, hexs(p, 0), hexs(p, 1),
           hexs(p, 2), hexs(p, 3), hexs(p, 4), hexs(p, 5), hexs(p, 6),
           hexs(p, 7));
  }
  first_few = std::min(3, int(cell_face_neighbors.num_rows));
  for (int c = 0; c < first_few; c++) {
    printf("cell %d face neighbors: ", c);
    auto neighbors = cell_face_neighbors(c);
    for (int i = 0; i < neighbors.size(); i++) {
      auto n = neighbors(i);
      printf("%d ", n);
    }
    printf("\n");
  }

  first_few = std::min(3, face_to_cell.extent_int(0));
  for (int f = 0; f < first_few; f++) {
    printf("face %d %d\n", face_to_cell(f, 0), face_to_cell(f, 1));
  }
}

template <typename Space>
typename vul::Grid<Space>::template Vec2D<int>
vul::Grid<Space>::getCellArray(vul::CellType type) {
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

template <typename Space> int vul::Grid<Space>::typeLength(vul::CellType type) {
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
template <typename Space> int vul::Grid<Space>::cellLength(int cell_id) const {
  return typeLength(cellType(cell_id));
}
template <typename Space>
vul::CellType vul::Grid<Space>::cellType(int cell_id) const {
  auto pair = cellIdToTypeAndIndexPair(cell_id);
  return pair.first;
}

template <typename Space> void vul::Grid<Space>::buildFaceNeighbors() {
  Kokkos::Profiling::pushRegion("buildFaceNeighbors");
  std::vector<std::vector<int>> neighbors(numCells());

  Kokkos::parallel_for(
      "buildFaceNeighbors", HostPolicy(0, numCells()), [&](int cell_id) {
        auto type = cellType(cell_id);
        std::vector<int> cell(cellLength(cell_id));
        getCell(cell_id, cell.data());
        auto candidates    = getNodeNeighborsOfCell(cell, cell_id);
        neighbors[cell_id] = getFaceNeighbors(type, cell, candidates);
      });

  cell_face_neighbors = CompressedRowGraph<vul::Host>(neighbors);
  Kokkos::Profiling::popRegion();
}

template <typename Space> void vul::Grid<Space>::buildFaces() {
  if (not std::is_same<Space, vul::Host>::value) {
    VUL_ASSERT((std::is_same<Space, vul::Host>::value),
               "Cannot build faces using a device grid");
  }
  Kokkos::Profiling::pushRegion("buildFaces");
  printf("Building node to cell adjacency:\n");
  {
    auto n2c_stl = buildNodeToCell();
    auto c2n_stl = buildCellToNode(n2c_stl);
    node_to_cell = CompressedRowGraph<Space>(n2c_stl);
    cell_to_node = CompressedRowGraph<Space>(c2n_stl);
  }
  buildFaceNeighbors();

  {
    // This math for num faces doesn't work in parallel if there are volume
    // cells that don't have face neighbors on rank.
    int num_faces = 6 * numHexs() + 5 * numPyramids() + 5 * numPrisms() +
                    4 * numTets() + numTris() + numQuads();
    num_faces /= 2;
    face_to_cell   = FaceToCells(NoInit("face_to_cell"), num_faces);
    face_to_nodes  = FaceToNodes(NoInit("face_to_nodes"), num_faces);
    face_area      = FaceArea(NoInit("face_area"), num_faces);
    face_centroids = PointVector<double>(NoInit("face_centroids"), num_faces);
  }

  int next_face = 0;
  std::vector<int> cell_nodes;
  std::vector<int> face_nodes;
  for (int c = 0; c < numCells(); c++) {
    int num_faces      = int(cell_face_neighbors(c).size());
    auto [type, index] = cellIdToTypeAndIndexPair(c);
    getCell(c, cell_nodes);
    Cell cell(type, cell_nodes);
    for (int face_number = 0; face_number < num_faces; face_number++) {
      int neighbor = cell_face_neighbors(c)(face_number);
      if (c < neighbor) {
        face_to_cell(next_face, 0)  = c;
        face_to_cell(next_face, 1)  = neighbor;
        Point<double> area          = calcFaceArea(cell.face(face_number));
        face_area(next_face, 0)     = area.x;
        face_area(next_face, 1)     = area.y;
        face_area(next_face, 2)     = area.z;
        auto face_nodes             = cell.face(face_number);
        face_to_nodes(next_face, 0) = face_nodes[0];
        face_to_nodes(next_face, 1) = face_nodes[1];
        face_to_nodes(next_face, 2) = face_nodes[2];
        if (face_nodes.size() == 4) {
          face_to_nodes(next_face, 3) = face_nodes[3];
        } else {
          face_to_nodes(next_face, 3) = -1; // quad
        }
        auto centroid                = calcFaceCentroid(cell.face(face_number));
        face_centroids(next_face, 0) = centroid.x;
        face_centroids(next_face, 1) = centroid.y;
        face_centroids(next_face, 2) = centroid.z;
        next_face++;
      }
    }
  }
  Kokkos::Profiling::popRegion();
}
template <typename Space>
void vul::Grid<Space>::getCell(int cell_id,
                               std::vector<int> &cell_nodes) const {
  cell_nodes.resize(cellLength(cell_id));
  getCell(cell_id, cell_nodes.data());
}
template <typename Space>
void vul::Grid<Space>::getCell(int cell_id, int *cell_nodes) const {
  auto [cell_type, cell_index] = cellIdToTypeAndIndexPair(cell_id);
  switch (cell_type) {
  case TET: {
    for (int i = 0; i < 4; i++) {
      cell_nodes[i] = tets(cell_index, i);
    }
    return;
  }
  case PYRAMID: {
    for (int i = 0; i < 5; i++) {
      cell_nodes[i] = pyramids(cell_index, i);
    }
    return;
  }
  case PRISM: {
    for (int i = 0; i < 6; i++) {
      cell_nodes[i] = prisms(cell_index, i);
    }
    return;
  }
  case HEX: {
    for (int i = 0; i < 8; i++) {
      cell_nodes[i] = hexs(cell_index, i);
    }
    return;
  }
  case TRI: {
    for (int i = 0; i < 3; i++) {
      cell_nodes[i] = tris(cell_index, i);
    }
    return;
  }
  case QUAD: {
    for (int i = 0; i < 4; i++) {
      cell_nodes[i] = quads(cell_index, i);
    }
    return;
  }
  default: {
    VUL_ASSERT(false,
               "Could not match cell type: " + std::to_string(cell_type));
  }
  }
}
template <typename Space>
std::vector<std::set<int>>
vul::Grid<Space>::buildCellToNode(const std::vector<std::set<int>> &n2c) {
  std::vector<std::set<int>> c2n(numCells());
  for (int node = 0; node < numPoints(); node++) {
    for (auto cell : n2c[node]) {
      c2n[cell].insert(node);
    }
  }
  return c2n;
}
template <typename Space>
std::vector<std::set<int>> vul::Grid<Space>::buildNodeToCell() {
  Kokkos::Profiling::pushRegion("buildNodeToCell");
  std::vector<std::set<int>> n2c(numPoints());
  std::vector<int> cell_nodes;
  cell_nodes.reserve(8);
  for (int c = 0; c < numCells(); c++) {
    getCell(c, cell_nodes);
    for (auto n : cell_nodes) {
      n2c[n].insert(c);
    }
  }
  Kokkos::Profiling::popRegion();
  return n2c;
}
template <typename Space> vul::Cell vul::Grid<Space>::cell(int cell_id) const {
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
template <typename Space>
std::vector<int>
vul::Grid<Space>::getNodeNeighborsOfCell(const std::vector<int> &cell_nodes,
                                         int cell_id) {
  if (not std::is_same<Space, vul::Host>::value) {
    VUL_ASSERT((std::is_same<Space, vul::Host>::value),
               "Cannot getNodeNeighborsOfCell using a device grid");
  }
  int neighbor_count = 0;
  for (int node : cell_nodes) {
    int row_length = node_to_cell.rows(node + 1) - node_to_cell.rows(node);
    neighbor_count += row_length;
  }

  std::vector<int> neighbors;
  neighbors.reserve(neighbor_count);
  for (int node : cell_nodes) {
    for (int i = node_to_cell.rows(node); i < node_to_cell.rows(node + 1);
         i++) {
      int c = node_to_cell.cols(i);
      if (c != cell_id and not isIn(neighbors, c)) {
        neighbors.push_back(c);
      }
    }
  }
  neighbors.shrink_to_fit();
  return neighbors;
}
template <typename Space>
std::vector<int>
vul::Grid<Space>::getFaceNeighbors(vul::CellType type,
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
template <typename Space>
int vul::Grid<Space>::findFaceNeighbor(const std::vector<int> &candidates,
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
template <typename Space>
bool vul::Grid<Space>::cellContainsFace(const std::vector<int> &cell,
                                        const std::vector<int> &face_nodes) {
  for (auto &id : face_nodes)
    if (std::find(cell.begin(), cell.end(), id) == cell.end())
      return false;
  return true;
}
template <typename Space>
vul::Point<double>
vul::Grid<Space>::calcFaceArea(const std::vector<int> &face_nodes) const {
  if (not std::is_same<Space, vul::Host>::value) {
    VUL_ASSERT((std::is_same<Space, vul::Host>::value),
               "Cannot calcFaceArea using a device grid");
  }
  std::array<Point<double>, 4> face_points;
  bool is_quad = face_nodes.size() == 4;
  for (int i = 0; i < 4; i++) {
    if (i == 3 and not is_quad)
      break; // break out for triangles;
    int n = face_nodes[i];
    Point<double> p;
    p.x            = points(n, 0);
    p.y            = points(n, 1);
    p.z            = points(n, 2);
    face_points[i] = p;
  }

  Point<double> u = face_points[1] - face_points[0];
  Point<double> v = face_points[2] - face_points[0];

  Point<double> area = u.cross(v) * 0.5;

  if (is_quad) {
    u    = face_points[0] - face_points[3];
    v    = face_points[2] - face_points[3];
    area = area + u.cross(v) * 0.5;
  }

  return area;
}
template <typename Space>
vul::Point<double>
vul::Grid<Space>::calcFaceCentroid(const std::vector<int> &face_nodes) const {
  if (not std::is_same<Space, vul::Host>::value) {
    VUL_ASSERT((std::is_same<Space, vul::Host>::value),
               "Cannot calcFaceArea using a device grid");
  }
  Point<double> centroid = {0.0, 0.0, 0.0};
  bool is_quad           = face_nodes.size() == 4;
  for (int i = 0; i < 4; i++) {
    if (i == 3 and not is_quad)
      break; // break out for triangles;
    int n    = face_nodes[i];
    auto p   = getPoint(n);
    centroid = centroid + p;
  }
  if (is_quad)
    centroid = centroid * 0.25;
  else
    centroid = centroid * (1.0 / 3.0);
  return centroid;
}
template <typename Space> void vul::Grid<Space>::computeCellVolumes() {
  int offset = 0;
  if (not std::is_same<Space, vul::Host>::value) {
    VUL_ASSERT((std::is_same<Space, vul::Host>::value),
               "Cannot compute cell volumes using a device grid");
  }
  cell_volume = Vec1D<double>("cell_volume", numCells());
  for (int t = 0; t < numTets(); t++) {
    double vol              = computeTetVolume(t);
    cell_volume(t + offset) = vol;
  }
  offset += numTets();

  for (int p = 0; p < numPyramids(); p++) {
    double vol              = computePyramidVolume(p);
    cell_volume(p + offset) = vol;
  }
  offset += numPyramids();
  for (int p = 0; p < numPrisms(); p++) {
    double vol              = computePrismVolume(p);
    cell_volume(p + offset) = vol;
  }
  offset += numPrisms();
  for (int p = 0; p < numHexs(); p++) {
    double vol              = computeHexVolume(p);
    cell_volume(p + offset) = vol;
  }
  offset += numHexs();
  for (int p = 0; p < numTris(); p++) {
    cell_volume(p + offset) = 0.0;
  }
  offset += numTris();
  for (int p = 0; p < numQuads(); p++) {
    cell_volume(p + offset) = 0.0;
  }
}
template <typename Space>
vul::Point<double> vul::Grid<Space>::getPoint(int n) const {
  return Point<double>{points(n, 0), points(n, 1), points(n, 2)};
}
template <typename Space>
vul::Point<double> vul::Grid<Space>::getCellCentroid(int c) const {
  return Point<double>{cell_centroids(c, 0), cell_centroids(c, 1),
                       cell_centroids(c, 2)};
}
template <typename Space>
vul::Point<double> vul::Grid<Space>::getFaceCentroid(int f) const {
  return Point<double>{face_centroids(f, 0), face_centroids(f, 1),
                       face_centroids(f, 2)};
}
template <typename Space> double vul::Grid<Space>::computeTetVolume(int t) {
  auto a   = getPoint(tets(t, 0));
  auto b   = getPoint(tets(t, 1));
  auto c   = getPoint(tets(t, 2));
  auto d   = getPoint(tets(t, 3));
  auto vol = computeTetVolume(a, b, c, d);
  return vol;
}
template <typename Space> double vul::Grid<Space>::computePyramidVolume(int p) {
  auto a   = getPoint(pyramids(p, 0));
  auto b   = getPoint(pyramids(p, 1));
  auto c   = getPoint(pyramids(p, 2));
  auto d   = getPoint(pyramids(p, 3));
  auto e   = getPoint(pyramids(p, 4));
  auto z   = (a + b + c + d) * 0.25;
  auto vol = computeTetVolume(a, b, z, e);
  vol += computeTetVolume(b, c, z, e);
  vol += computeTetVolume(c, d, z, e);
  vol += computeTetVolume(d, a, z, e);
  return vol;
}
template <typename Space> double vul::Grid<Space>::computePrismVolume(int p) {
  auto a = getPoint(prisms(p, 0));
  auto b = getPoint(prisms(p, 1));
  auto c = getPoint(prisms(p, 2));
  auto d = getPoint(prisms(p, 3));
  auto e = getPoint(prisms(p, 4));
  auto f = getPoint(prisms(p, 5));

  auto centroid = (a + b + c + d + e + f) * (1.0 / 6.0);

  double vol = computeTetVolume(a, b, c, centroid);
  vol += computeTetVolume(b, e, c, centroid);
  vol += computeTetVolume(c, e, f, centroid);

  vol += computeTetVolume(c, f, a, centroid);
  vol += computeTetVolume(a, f, d, centroid);

  vol += computeTetVolume(a, d, b, centroid);
  vol += computeTetVolume(d, e, b, centroid);

  vol += computeTetVolume(d, f, e, centroid);
  return vol;
}
template <typename Space> double vul::Grid<Space>::computeHexVolume(int p) {
  auto a     = getPoint(hexs(p, 0));
  auto b     = getPoint(hexs(p, 1));
  auto c     = getPoint(hexs(p, 2));
  auto d     = getPoint(hexs(p, 3));
  auto e     = getPoint(hexs(p, 4));
  auto f     = getPoint(hexs(p, 5));
  auto g     = getPoint(hexs(p, 6));
  auto h     = getPoint(hexs(p, 7));
  double vol = 0.0;

  auto centroid = (a + b + c + d + e + f + g + h) * (1.0 / 8.0);

  vol += computeTetVolume(a, b, d, centroid);
  vol += computeTetVolume(b, c, d, centroid);

  vol += computeTetVolume(a, e, b, centroid);
  vol += computeTetVolume(e, f, b, centroid);

  vol += computeTetVolume(b, f, c, centroid);
  vol += computeTetVolume(c, f, g, centroid);

  vol += computeTetVolume(d, g, h, centroid);
  vol += computeTetVolume(d, c, g, centroid);

  vol += computeTetVolume(d, h, e, centroid);
  vol += computeTetVolume(d, e, a, centroid);

  vol += computeTetVolume(e, h, f, centroid);
  vol += computeTetVolume(f, h, g, centroid);

  return vol;
}
template <typename Space>
double vul::Grid<Space>::computeTetVolume(const Point<double> &a,
                                          const Point<double> &b,
                                          const Point<double> &c,
                                          const Point<double> &d) {

  auto v1 = a - d;
  auto v2 = b - d;
  auto v3 = c - d;
  auto v  = v2.cross(v3);
  return -v1.dot(v) / 6.0;
}
template <typename Space>
int vul::Grid<Space>::getVulCellIdFromInfId(int inf_id) const {
  int num_surface = numTris() + numQuads();
  if (inf_id < num_surface) {
    return inf_id + numVolumeCells();
  } else {
    return inf_id - num_surface;
  }
}
template <typename Space>
void vul::Grid<Space>::setCartesianPoints(int n_cells_x, int n_cells_y,
                                          int n_cells_z) {
  if (not std::is_same<Space, vul::Host>::value) {
    VUL_ASSERT((std::is_same<Space, vul::Host>::value),
               "Cannot setCartesianPoints using a device grid");
  }
  CartBlock block(n_cells_x, n_cells_y, n_cells_z);
  for (int n = 0; n < points.extent_int(0); n++) {
    auto p = block.getPoint(n);
    for (int i = 0; i < 3; i++)
      points(n, i) = p.pos[i];
  }
}
template <typename Space>
void vul::Grid<Space>::setCartesianCells(int n_cells_x, int n_cells_y,
                                         int n_cells_z) {
  if (not std::is_same<Space, vul::Host>::value) {
    VUL_ASSERT((std::is_same<Space, vul::Host>::value),
               "Cannot setCartesianCells using a device grid");
  }
  CartBlock block(n_cells_x, n_cells_y, n_cells_z);
  // set hexs
  for (int c = 0; c < hexs.extent_int(0); c++) {
    auto h = block.getNodesInCell(c);
    for (int i = 0; i < 8; i++) {
      hexs(c, i) = h[i];
    }
  }

  // set quads
  // bottom
  int quad_index = 0;
  for (int i = 0; i < n_cells_x; i++) {
    for (int j = 0; j < n_cells_y; j++) {
      quads(quad_index, 0)    = block.convert_ijk_ToNodeId(i, j, 0);
      quads(quad_index, 1)    = block.convert_ijk_ToNodeId(i, j + 1, 0);
      quads(quad_index, 2)    = block.convert_ijk_ToNodeId(i + 1, j + 1, 0);
      quads(quad_index, 3)    = block.convert_ijk_ToNodeId(i + 1, j, 0);
      quad_tags(quad_index++) = 1;
    }
  }

  // top
  int num_z = n_cells_z;
  for (int i = 0; i < n_cells_x; i++) {
    for (int j = 0; j < n_cells_y; j++) {
      quads(quad_index, 0)    = block.convert_ijk_ToNodeId(i, j, num_z);
      quads(quad_index, 1)    = block.convert_ijk_ToNodeId(i + 1, j, num_z);
      quads(quad_index, 2)    = block.convert_ijk_ToNodeId(i + 1, j + 1, num_z);
      quads(quad_index, 3)    = block.convert_ijk_ToNodeId(i, j + 1, num_z);
      quad_tags(quad_index++) = 6;
    }
  }

  // front
  for (int j = 0; j < n_cells_y; j++) {
    for (int k = 0; k < n_cells_z; k++) {
      quads(quad_index, 0)    = block.convert_ijk_ToNodeId(0, j, k);
      quads(quad_index, 1)    = block.convert_ijk_ToNodeId(0, j, k + 1);
      quads(quad_index, 2)    = block.convert_ijk_ToNodeId(0, j + 1, k + 1);
      quads(quad_index, 3)    = block.convert_ijk_ToNodeId(0, j + 1, k);
      quad_tags(quad_index++) = 2;
    }
  }

  // back
  int num_x = n_cells_x;
  for (int j = 0; j < n_cells_y; j++) {
    for (int k = 0; k < n_cells_z; k++) {
      quads(quad_index, 0)    = block.convert_ijk_ToNodeId(num_x, j, k);
      quads(quad_index, 1)    = block.convert_ijk_ToNodeId(num_x, j + 1, k);
      quads(quad_index, 2)    = block.convert_ijk_ToNodeId(num_x, j + 1, k + 1);
      quads(quad_index, 3)    = block.convert_ijk_ToNodeId(num_x, j, k + 1);
      quad_tags(quad_index++) = 4;
    }
  }

  // right
  for (int i = 0; i < n_cells_x; i++) {
    for (int k = 0; k < n_cells_z; k++) {
      quads(quad_index, 0)    = block.convert_ijk_ToNodeId(i, 0, k);
      quads(quad_index, 1)    = block.convert_ijk_ToNodeId(i + 1, 0, k);
      quads(quad_index, 2)    = block.convert_ijk_ToNodeId(i + 1, 0, k + 1);
      quads(quad_index, 3)    = block.convert_ijk_ToNodeId(i, 0, k + 1);
      quad_tags(quad_index++) = 3;
    }
  }

  // left
  int num_y = n_cells_y;
  for (int i = 0; i < n_cells_x; i++) {
    for (int k = 0; k < n_cells_z; k++) {
      quads(quad_index, 0)    = block.convert_ijk_ToNodeId(i, num_y, k);
      quads(quad_index, 1)    = block.convert_ijk_ToNodeId(i, num_y, k + 1);
      quads(quad_index, 2)    = block.convert_ijk_ToNodeId(i + 1, num_y, k + 1);
      quads(quad_index, 3)    = block.convert_ijk_ToNodeId(i + 1, num_y, k);
      quad_tags(quad_index++) = 5;
    }
  }
}
template <typename Space> void vul::Grid<Space>::computeCellCentroids() {
  if (not std::is_same<Space, vul::Host>::value) {
    VUL_ASSERT((std::is_same<Space, vul::Host>::value),
               "Cannot compute cell centroids using a device grid");
  }
  cell_centroids = PointVector<double>("cell_centroids", numCells());
  std::vector<int> cell_nodes;
  for (int c = 0; c < numCells(); c++) {
    getCell(c, cell_nodes);
    Point<double> centroid = {0, 0, 0};
    for (int n : cell_nodes) {
      auto p   = getPoint(n);
      centroid = centroid + p;
    }
    centroid = centroid * (1.0 / double(cell_nodes.size()));
    for (int i = 0; i < 3; i++) {
      cell_centroids(c, i) = centroid.pos[i];
    }
  }
}
template <typename Space>
vul::Point<double> vul::Grid<Space>::getCellCentroid(int i) {
  vul::Point<double> p;
  p.x = cell_centroids(i, 0);
  p.y = cell_centroids(i, 1);
  p.z = cell_centroids(i, 2);
  return p;
}
template <typename Space>
KOKKOS_FUNCTION Kokkos::pair<vul::CellType, int>
vul::Grid<Space>::cellIdToTypeAndIndexPair(int cell_id) const {
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
  // VUL_ASSERT(false, "Could not find type of cell_id " +
  // std::to_string(orig_cell_id));
  return {TRI, -1}; // return bad data.  Hopefully we find it quickly, we
  // can't assert on device...
}

#pragma once
#include <vector>
#include "Macros.h"

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


inline vul::Cell::Cell(vul::CellType type, const std::vector<int> &nodes)
    : _type(type), cell_nodes(nodes) {}

inline std::vector<int> vul::Cell::face(int i) const {
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
inline int vul::Cell::numFaces() {
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

inline vul::Cell::Cell(vul::CellType type, const int *nodes) : _type(type) {
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
  default:
    // put a gpu compatible error message here.
    return;
  }
}


}
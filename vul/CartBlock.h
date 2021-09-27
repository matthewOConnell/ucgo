#pragma once
#include "Point.h"
#include "Macros.h"

class CartBlock {
public:
  CartBlock(int ncells_x, int ncells_y, int ncells_z)
      : kx(ncells_x), ky(ncells_y), kz(ncells_z) {
    // always assume a unit cube
    dx = 1.0 / double(ncells_x);
    dy = 1.0 / double(ncells_y);
    dz = 1.0 / double(ncells_z);
  }

  inline vul::Point<double> getPoint(int n) const {
    int i, j, k;
    convertNodeIdTo_ijk(n, i, j, k);
    return vul::Point<double>{dx * i, dy * j, dz * k};
  }

  inline void convertNodeIdTo_ijk(int node_id, int &i, int &j, int &k) const {
    int nx = kx + 1;
    int ny = ky + 1;
    k      = node_id / (nx * ny);
    j      = (node_id - k * nx * ny) / nx;
    i      = node_id - k * nx * ny - j * nx;
  }
  inline void convertCellIdTo_ijk(int cell_id, int &i, int &j, int &k) const {
    k = cell_id / (kx * ky);
    j = (cell_id - k * kx * ky) / kx;
    i = cell_id - k * kx * ky - j * kx;
  }
  inline int convert_ijk_ToNodeId(int i, int j, int k) const {
    int nx = kx + 1;
    int ny = ky + 1;
    return i + j * nx + k * nx * ny;
  }

  inline vul::StaticIntArray<8> getNodesInCell(int id) const {
    int i, j, k;
    convertCellIdTo_ijk(id, i, j, k);

    vul::StaticIntArray<8> cellNodes;
    cellNodes[0] = convert_ijk_ToNodeId(i, j, k);
    cellNodes[1] = convert_ijk_ToNodeId(i + 1, j, k);
    cellNodes[2] = convert_ijk_ToNodeId(i + 1, j + 1, k);
    cellNodes[3] = convert_ijk_ToNodeId(i, j + 1, k);
    cellNodes[4] = convert_ijk_ToNodeId(i, j, k + 1);
    cellNodes[5] = convert_ijk_ToNodeId(i + 1, j, k + 1);
    cellNodes[6] = convert_ijk_ToNodeId(i + 1, j + 1, k + 1);
    cellNodes[7] = convert_ijk_ToNodeId(i, j + 1, k + 1);
    return cellNodes;
  }

private:
  int kx, ky, kz;
  double dx, dy, dz;
};

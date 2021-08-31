#pragma once
#include <string>

namespace vul {
class Grid {
  public:
    enum CellType {TRI, QUAD, TET, PYRAMID, PRISM, HEX};
    Grid(std::string filename);

    int count(CellType type) const;
  private:
};
}


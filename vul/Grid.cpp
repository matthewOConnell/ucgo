#include "Grid.h"
#include "Macros.h"

vul::Grid::Grid(std::string filename) {
    FILE* fp = fopen(filename.c_str(), "r");
    VUL_ASSERT(fp != nullptr, "Could not open file: " + filename);

    fclose(fp);
}
int vul::Grid::count(vul::Grid::CellType type) const {
    return 0;
}

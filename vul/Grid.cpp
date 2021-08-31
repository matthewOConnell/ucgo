#include "Grid.h"

#include "Macros.h"

vul::Grid::Grid(std::string filename) {
    FILE* fp = fopen(filename.c_str(), "r");
    VUL_ASSERT(fp != nullptr, "Could not open file: " + filename);

    int nnodes, ntri, nquad, ntet, npyramid, nprism, nhex;
    fread(&nnodes, sizeof(int), 1, fp);
    fread(&ntri, sizeof(int), 1, fp);
    fread(&nquad, sizeof(int), 1, fp);
    fread(&ntet, sizeof(int), 1, fp);
    fread(&npyramid, sizeof(int), 1, fp);
    fread(&nprism, sizeof(int), 1, fp);
    fread(&nhex, sizeof(int), 1, fp);

    points = Vec2D("points", nnodes, 3);
    tris = Vec2D("tris", ntri, 3);
    quads = Vec2D("quads", nquad, 4);
    pyramids = Vec2D("pyramids", npyramid, 5);
    prisms = Vec2D("prisms", nprism, 6);
    hexs = Vec2D("hexs", nhex, 8);

    fclose(fp);
}
int vul::Grid::count(vul::Grid::CellType type) const {
    switch (type) {
        case TRI: return tris.extent_int(0);
        case QUAD: return quads.extent_int(0);
        case TET: return tets.extent_int(0);
        case PYRAMID: return pyramids.extent_int(0);
        case PRISM: return prisms.extent_int(0);
        case HEX: return hexs.extent_int(0);
        default:
            throw std::logic_error("Unknown type requested in count" + std::to_string(type));
    }
    return 0;
}

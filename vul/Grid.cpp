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

    tri_tags = Vec1D("tri_tags", ntri);
    quad_tags = Vec1D("quad_tags", nquad);

    readPoints(fp);

    fclose(fp);
}

void vul::Grid::readPoints(FILE* fp) {
    int nnodes = points.h_view.extent_int(0);
    std::vector<double> points_buffer(nnodes*3);
    fread(points_buffer.data(), sizeof(double), 3*nnodes, fp);
    for(int p = 0; p < nnodes; p++){
        points.h_view(p, 0) = points_buffer[3*p+0];
        points.h_view(p, 1) = points_buffer[3*p+1];
        points.h_view(p, 2) = points_buffer[3*p+2];
    }
    Kokkos::deep_copy(points.d_view, points.h_view);
}

void vul::Grid::readCells(FILE* fp) {
    std::vector<int> buffer;
    int ncells;
    int length;

    ncells = tris.extent_int(0);
    length = 3;
    buffer.resize(ncells*length);
    fread(buffer.data(), buffer.size(), sizeof(int), fp);
    for(int c = 0; c < ncells; c++){
        for(int i= 0; i < length; i++){
            tris.h_view(c, i) = buffer[length*c+i];
        }
    }
    Kokkos::deep_copy(tris.d_view, tris.h_view);

    ncells = quads.extent_int(0);
    length = 4;
    buffer.resize(ncells*length);
    fread(buffer.data(), buffer.size(), sizeof(int), fp);
    for(int c = 0; c < ncells; c++){
        for(int i= 0; i < length; i++){
            quads.h_view(c, i) = buffer[length*c+i];
        }
    }
    Kokkos::deep_copy(quads.d_view, quads.h_view);

    ncells = tri_tags.extent_int(0);
    buffer.resize(ncells);
    fread(buffer.data(), buffer.size(), sizeof(int), fp);
    for(int c = 0; c < ncells; c++){
        tri_tags.h_view(c) = buffer[c];
    }
    Kokkos::deep_copy(tri_tags.d_view, tri_tags.h_view);

    ncells = quad_tags.extent_int(0);
    buffer.resize(ncells);
    fread(buffer.data(), buffer.size(), sizeof(int), fp);
    for(int c = 0; c < ncells; c++){
        quad_tags.h_view(c) = buffer[c];
    }
    Kokkos::deep_copy(quad_tags.d_view, quad_tags.h_view);

    ncells = tets.extent_int(0);
    length = 4;
    buffer.resize(ncells*length);
    fread(buffer.data(), buffer.size(), sizeof(int), fp);
    for(int c = 0; c < ncells; c++){
        for(int i= 0; i < length; i++){
            tets.h_view(c, i) = buffer[length*c+i];
        }
    }
    Kokkos::deep_copy(tets.d_view, tets.h_view);

    ncells = pyramids.extent_int(0);
    length = 5;
    buffer.resize(ncells*length);
    fread(buffer.data(), buffer.size(), sizeof(int), fp);
    for(int c = 0; c < ncells; c++){
        for(int i= 0; i < length; i++){
            pyramids.h_view(c, i) = buffer[length*c+i];
        }
        //--- swap from AFLR to CGNS winding
        std::swap(pyramids.h_view(c, 2), pyramids.h_view(c, 4));
        std::swap(pyramids.h_view(c, 1), pyramids.h_view(c, 3));
    }
    Kokkos::deep_copy(pyramids.d_view, pyramids.h_view);

    ncells = prisms.extent_int(0);
    length = 6;
    buffer.resize(ncells*length);
    fread(buffer.data(), buffer.size(), sizeof(int), fp);
    for(int c = 0; c < ncells; c++){
        for(int i= 0; i < length; i++){
            prisms.h_view(c, i) = buffer[length*c+i];
        }
    }
    Kokkos::deep_copy(prisms.d_view, prisms.h_view);

    ncells = hexs.extent_int(0);
    length = 8;
    buffer.resize(ncells*length);
    fread(buffer.data(), buffer.size(), sizeof(int), fp);
    for(int c = 0; c < ncells; c++){
        for(int i= 0; i < length; i++){
            hexs.h_view(c, i) = buffer[length*c+i];
        }
    }
    Kokkos::deep_copy(hexs.d_view, hexs.h_view);

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

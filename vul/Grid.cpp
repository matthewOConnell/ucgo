#include "Macros.h"
#include "Grid.hpp"

namespace vul {
template class Grid<Host>;
template class Grid<Device>;

template void Grid<Host>::deep_copy(const Grid<Device>&);
template void Grid<Device>::deep_copy(const Grid<Host>&);
}

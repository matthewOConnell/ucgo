#include "Macros.h"
#include "Grid.hpp"

namespace vul {
template class Grid<Host>;
template class Grid<Device>;

template Grid<Host>::Grid(const Grid<Device>&);
template Grid<Device>::Grid(const Grid<Host>&);
}

#include "Macros.h"
#include "Grid.hpp"

template class vul::Grid<vul::Host>;
#ifdef KOKKOS_ENABLE_CUDA
template class vul::Grid<vul::Device>;
#endif


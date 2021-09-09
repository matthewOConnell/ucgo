#include <stdio.h>
#include <string>
#include <Kokkos_Core.hpp>
#include "Vulcan.h"

void solve(std::string filename) {
  printf("Inside solve\n");
  vul::Vulcan<5, 2> vulcan(filename);
  int num_iterations = 10;
  vulcan.solve(num_iterations);
}

int main(int num_args, const char* args[]) {
  Kokkos::InitArguments arguments;
  arguments.disable_warnings = true;
  Kokkos::initialize(arguments);

  std::ostringstream msg;
#if defined(__CUDACC__)
  Kokkos::Cuda::print_configuration(msg);
#endif
  std::cout << msg.str() << std::endl;


  std::string assets_dir = ASSETS_DIR;
  std::string filename   = assets_dir + "/";
  if(num_args == 2){
    filename += args[1];
  } else {
    filename += "shock.lb8.ugrid";
  }

  // It is best if all the kokkos objects live in a function call
  // and not in the main function.
  // So this "solve" method is just a wrapper to do everything
  solve(filename);

  Kokkos::finalize();
  return 0;
}

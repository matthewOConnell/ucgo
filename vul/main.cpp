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

void printLegalStatement(){
printf("Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.\n");
printf("Third Party Software:\nThis software calls the following third party software, which is subject to the terms and conditions of its licensor, as applicable at the time of licensing.  Third party software is not bundled with this software, but may be available from the licensor.  License hyperlinks are provided here for information purposes only:  Kokkos v3.0, 3-clause BSD license, https://github.com/kokkos/kokkos, under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this third-party software.\n");
printf("The Unstructured CFD graph operations miniapp framework is licensed under the Apache License, Version 2.0 (the \"License\"); you may not use this application except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. \n");

printf("Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.\n");
}

int main(int num_args, const char* args[]) {

    printLegalStatement();

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

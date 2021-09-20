#include "Vulcan.h"
#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <string>
#include <vector>

std::string getFilenameFromArgs(const std::vector<std::string> &args) {
  for (long i = 0; i < args.size(); i++) {
    auto a = args[i];
    if (a == "-f" or a == "--file")
      return args[i + 1];
  }
  return "";
}

std::array<int, 3> getDimensionsFromArgs(const std::vector<std::string> &args) {
  for (long i = 0; i < args.size(); i++) {
    auto a = args[i];
    if (a == "-g" or a == "--generate") {
      std::array<int, 3> num_cells;
      num_cells[0] = std::stoi(args[i + 1]);
      num_cells[1] = std::stoi(args[i + 2]);
      num_cells[2] = std::stoi(args[i + 3]);
      return num_cells;
    }
  }
  return {-1, -1, -1};
}

void solve(const std::vector<std::string> &args) {
  printf("Inside solve\n");
  auto filename = getFilenameFromArgs(args);
  std::unique_ptr<vul::Grid> grid;
  if (filename != "") {
    grid = std::make_unique<vul::Grid>(filename);
  } else {
    auto num_cells = getDimensionsFromArgs(args);
    if(num_cells[0] == -1) {
      printf("Error num cells is negative and grid filename is empty\n");
      exit(1);
    }
    grid = std::make_unique<vul::Grid>(num_cells[0], num_cells[1], num_cells[2]);
  }

  vul::Vulcan<5, 2> vulcan(*grid);
  Kokkos::Profiling::popRegion();
  int num_iterations = 10;
  Kokkos::Profiling::pushRegion("solve");
  vulcan.solve(num_iterations);
  Kokkos::Profiling::popRegion();
}

void printLegalStatement() {
  printf("Copyright 2021 United States Government as represented by the "
         "Administrator of the National Aeronautics and Space Administration. "
         "No copyright is claimed in the United States under Title 17, U.S. "
         "Code. All Other Rights Reserved.\n");
  printf(
      "Third Party Software:\nThis software calls the following third party "
      "software, which is subject to the terms and conditions of its licensor, "
      "as applicable at the time of licensing.  Third party software is not "
      "bundled with this software, but may be available from the licensor.  "
      "License hyperlinks are provided here for information purposes only:  "
      "Kokkos v3.0, 3-clause BSD license, https://github.com/kokkos/kokkos, "
      "under the terms of Contract DE-NA0003525 with NTESS, the U.S. "
      "Government retains certain rights in this third-party software.\n");
  printf("The Unstructured CFD graph operations miniapp framework is licensed "
         "under the Apache License, Version 2.0 (the \"License\"); you may not "
         "use this application except in compliance with the License. You may "
         "obtain a copy of the License at "
         "http://www.apache.org/licenses/LICENSE-2.0. \n");

  printf("Unless required by applicable law or agreed to in writing, software "
         "distributed under the License is distributed on an \"AS IS\" BASIS, "
         "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or "
         "implied. See the License for the specific language governing "
         "permissions and limitations under the License.\n");
}

int main(int num_args, const char *args[]) {

  printLegalStatement();

  Kokkos::InitArguments arguments;
  arguments.disable_warnings = true;
  Kokkos::initialize(arguments);

  std::ostringstream msg;
#if defined(__CUDACC__)
  Kokkos::Cuda::print_configuration(msg);
#endif
  std::cout << msg.str() << std::endl;
  Kokkos::Profiling::pushRegion("setup");

  std::string assets_dir = ASSETS_DIR;
  std::string filename   = assets_dir + "/";
  std::vector<std::string> command_line_args;
  for (int i = 1; i < num_args; i++) {
    auto a = args[i];
    command_line_args.push_back(a);
  }
  if (command_line_args.empty()) {
    // -- if no argument given.  Assume 10k grid
    command_line_args.push_back("-g");
    command_line_args.push_back("100");
    command_line_args.push_back("10");
    command_line_args.push_back("10");
  }

  // It is best if all the kokkos objects live in a function call
  // and not in the main function.
  // So this "solve" method is just a wrapper to do everything
  solve(command_line_args);

  Kokkos::finalize();
  return 0;
}

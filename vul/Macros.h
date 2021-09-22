//Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.
//Third Party Software:
//This software calls the following third party software, which is subject to the terms and conditions of its licensor, as applicable at the time of licensing.  Third party software is not bundled with this software, but may be available from the licensor.  License hyperlinks are provided here for information purposes only:  Kokkos v3.0, 3-clause BSD license, https://github.com/kokkos/kokkos, under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this third-party software.
//The Unstructured CFD graph operations miniapp platform is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. 
//Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
#pragma once
#include <string>
#include <stdexcept>
#include <Kokkos_Core.hpp>

#define VUL_ASSERT( boolean_statement, message) { if(not (boolean_statement)) {throw std::logic_error(std::string("ASSERT_FAILED: ") + std::string(message) + " at file: " + std::string(__FILE__) + " function: " + std::string(__func__) + " line: " + std::to_string(__LINE__) + std::string("\n"));}}

namespace vul {

using Host = Kokkos::DefaultHostExecutionSpace::memory_space;
using Device = Kokkos::DefaultExecutionSpace::memory_space;

template <typename View1, typename View2>
void force_copy(View1 to, View2 from) {
  using WriteSpace  = typename View1::memory_space;
  using ReadSpace   = typename View2::memory_space;
  using HostSpace   = Kokkos::DefaultHostExecutionSpace::memory_space;
  using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;
  // same to same
  if (std::is_same<ReadSpace, WriteSpace>::value) {
    Kokkos::deep_copy(to, from);
    return;
  }
  // host to device
  if (std::is_same<WriteSpace, DeviceSpace>::value) {
    auto mirror = create_mirror_view(to);
    Kokkos::deep_copy(mirror, from);
    Kokkos::deep_copy(to, mirror);
    return;
  }
  // device to host
  if (std::is_same<WriteSpace, HostSpace>::value) {
    auto mirror = create_mirror_view(from);
    Kokkos::deep_copy(mirror, from);
    Kokkos::deep_copy(to, mirror);
    return;
  }
  printf("You should not be here.\n");
}

}
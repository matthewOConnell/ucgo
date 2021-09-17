//Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.
//Third Party Software:
//This software calls the following third party software, which is subject to the terms and conditions of its licensor, as applicable at the time of licensing.  Third party software is not bundled with this software, but may be available from the licensor.  License hyperlinks are provided here for information purposes only:  Kokkos v3.0, 3-clause BSD license, https://github.com/kokkos/kokkos, under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this third-party software.
//The Unstructured CFD graph operations miniapp platform is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. 
//Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
#pragma once
namespace vul {
class PhysicalFlux {
public:
  template <size_t N, size_t NG>
  static StaticArray<N> inviscidFlux(const StaticArray<N> &q,
                              const StaticArray<NG> &qg,
                              const Point<double> &face_area) {
    int num_species = 1;
    const double *densities    = &q[0];
    const double &u            = q[1]/q[0];
    const double &v            = q[2]/q[0];
    const double &w            = q[3]/q[0];
    const double &total_energy = q[4];
    const double pressure = qg[1];

    double density = 0.0;
    for (size_t i = 0; i < num_species; ++i)
      density += densities[i];

    double unorm = u * face_area.x + v * face_area.y + w * face_area.z;

    StaticArray<N> physical_fluxes;
    for (int s = 0; s < num_species; ++s)
      physical_fluxes[s] = unorm * densities[s];

    physical_fluxes[1] = unorm * density * u + face_area.x * pressure;
    physical_fluxes[2] = unorm * density * v + face_area.y * pressure;
    physical_fluxes[3] = unorm * density * w + face_area.z * pressure;
    physical_fluxes[4] = unorm * (total_energy + pressure);

    return physical_fluxes;
  }
};

} // namespace vul

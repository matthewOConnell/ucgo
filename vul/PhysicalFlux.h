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
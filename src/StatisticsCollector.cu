#include "StatisticsCollector.cuh"

namespace cfd {
StatisticsCollector::StatisticsCollector(Parameter &parameter, const Mesh &_mesh, const std::vector<Field> &_field)
    : if_collect_statistics{parameter.get_bool("if_collect_statistics")},
      start_iter{parameter.get_int("start_collect_statistics_iter")},
      mesh{_mesh}, field{_field} {
  if (!if_collect_statistics)
    return;

}

__global__ void collect_basic_variable_statistics(DZone *zone) {
  const integer extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  const auto &bv = zone->bv;
  auto &s12M = zone->sum12Moment;
  auto &s34M = zone->sum34Moment;
  auto &srss = zone->sumReynoldsShearPart;

  // 0-5: rho, u, v, w, p, t
  // 6-11: rho*rho, u*u, v*v, w*w, p*p, t*t
#pragma unroll
  for (integer l = 0; l < 6; ++l) {
    s12M(i, j, k, l) += bv(i, j, k, l);
    s12M(i, j, k, l + 6) += bv(i, j, k, l) * bv(i, j, k, l);
    s34M(i, j, k, l) += bv(i, j, k, l) * bv(i, j, k, l) * bv(i, j, k, l);
    s34M(i, j, k, l + 6) += bv(i, j, k, l) * bv(i, j, k, l) * bv(i, j, k, l) * bv(i, j, k, l);
  }

  // The shear part of the reynolds stress
  srss(i, j, k, 0) += bv(i, j, k, 1) * bv(i, j, k, 2); // u*v
  srss(i, j, k, 1) += bv(i, j, k, 1) * bv(i, j, k, 3); // u*w
  srss(i, j, k, 2) += bv(i, j, k, 2) * bv(i, j, k, 3); // v*w
}

__global__ void collect_species_statistics(DZone *zone, DParameter *param) {
  const integer extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  const auto &sv = zone->sv;
  auto &sYk = zone->sumYk;

  for (integer l = 0; l < param->n_spec; ++l) {
    sYk(i, j, k, l) += sv(i, j, k, l);
  }
}
} // cfd
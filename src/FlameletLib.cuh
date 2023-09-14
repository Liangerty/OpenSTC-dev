#pragma once

#include "Define.h"
#include <vector>
#include "Parameter.h"
#include "gxl_lib/Matrix.hpp"
#include "gxl_lib/Array.hpp"
#include "DParameter.cuh"

namespace cfd {

struct FlameletLib {
  integer n_spec{0};
  integer n_z = 0, n_zPrime = 0, n_chi = 0;
  std::vector<real> z, chi, fz, dz, ffz, zi;
  real fzst = 0;
  gxl::MatrixDyn<real> zPrime, chi_min, chi_max;
  gxl::MatrixDyn<integer> chi_min_j, chi_max_j;
  gxl::Array3D<real> chi_ave;
  gxl::VectorField3D<real> yk;

  explicit FlameletLib(const Parameter &parameter);

private:
  void read_ACANS_flamelet(const Parameter &parameter);
};

struct DZone;

__device__ void flamelet_source(cfd::DZone *zone, integer i, integer j, integer k, DParameter *param);

} // cfd
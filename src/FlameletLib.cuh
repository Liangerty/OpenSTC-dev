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

__device__ void
compute_massFraction_from_MixtureFraction(cfd::DZone *zone, integer i, integer j, integer k, DParameter *param,
                                          real *yk_ave);

__device__ void
interpolate_scalar_dissipation_rate_with_given_z_zPrime(real chi_ave, integer n_spec, integer i_z, integer i_zPrime,
                                                        DParameter *param, real *yk);

__device__ int2
find_chi_range(const ggxl::Array3D<real> &chi_ave, real chi, integer i_z, integer i_zPrime, integer n_chi);

__global__ void update_n_fl_step(DParameter *param);

} // cfd
#pragma once

#include "Define.h"
#include "Mesh.h"

namespace cfd {
struct DZone;
struct DParameter;

template<MixtureModel mix_model>
void Roe_compute_inviscid_flux(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var,
                               const Parameter &parameter);

template<MixtureModel mix_model>
__global__ void
Roe_compute_inviscid_flux_1D(cfd::DZone *zone, integer direction, integer max_extent, DParameter *param);

template<MixtureModel mix_model>
__device__ void
Roe_compute_half_point_flux(DZone *zone, real *pv, integer tid, DParameter *param, real *fc, real *metric,
                            const real *jac, const real *entropy_fix_delta, integer direction);

template<MixtureModel mix_model>
__global__ void
compute_entropy_fix_delta(cfd::DZone *zone, DParameter *param);

template<MixtureModel mixtureModel>
__device__ void
compute_half_sum_left_right_flux(const real *pv_l, const real *pv_r, DParameter *param, const real *jac,
                                 const real *metric,
                                 integer i_shared, integer direction, real *fc);

}
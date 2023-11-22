#pragma once

#include "Define.h"
#include "Mesh.h"
#include "gxl_lib/Array.hpp"

namespace cfd {
struct DZone;
struct DParameter;

template<MixtureModel mix_model>
void AWENO_LF(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var, const Parameter &parameter);

template<MixtureModel mix_model>
void AWENO_HLLC(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var, const Parameter &parameter);

template<MixtureModel mix_model>
__global__ void LFPart1D(cfd::DZone *zone, integer direction, integer max_extent, DParameter *param);

template<MixtureModel mix_model>
__global__ void HLLCPart1D(cfd::DZone *zone, integer direction, integer max_extent, DParameter *param);

template<MixtureModel mix_model>
__device__ void
AWENO_interpolation(const real *cv, real *pv_l, real *pv_r, integer idx_shared, integer n_var, const real *metric,
                    DParameter *param);

__device__ double2 WENO5(const real *L, const real *cv, integer n_var, integer i_shared, integer l_row);

template<MixtureModel mix_model>
__global__ void CDSPart1D(cfd::DZone *zone, integer direction, integer max_extent, DParameter *param);
// Implementations
}
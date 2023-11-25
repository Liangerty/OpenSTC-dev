#pragma once

#include "Define.h"
#include "DParameter.cuh"

namespace cfd {
template<MixtureModel mix_model>
__device__ void
compute_ausmPlus_flux(const real *pv_l, const real *pv_r, DParameter *param, integer tid, const real *metric,
                      const real *jac, real *fc, integer i_shared);

}
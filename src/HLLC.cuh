#pragma once

#include "Define.h"
#include "Mesh.h"

namespace cfd {
struct DZone;
struct DParameter;

template<MixtureModel mix_model>
__device__ void
compute_hllc_flux(const real *pv_l, const real *pv_r, DParameter *param, integer tid, const real *metric,
                  const real *jac, real *fc, integer i_shared);

}
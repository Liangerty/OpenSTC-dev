#include "DPLUR.cuh"

namespace cfd {
__global__ void convert_dq_back_to_dqDt(DZone *zone, const DParameter *param) {
  const integer extent[3]{zone->mx, zone->my, zone->mz};
  const integer i = blockDim.x * blockIdx.x + threadIdx.x;
  const integer j = blockDim.y * blockIdx.y + threadIdx.y;
  const integer k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  const real dt_local = zone->dt_local(i, j, k);
  auto &dq = zone->dq;
  for (integer l = 0; l < param->n_var; ++l) {
    dq(i, j, k, l) /= dt_local;
  }
}

}
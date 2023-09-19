#include "DataCommunication.cuh"

__global__ void cfd::setup_data_to_be_sent(cfd::DZone *zone, integer i_face, real *data, const DParameter* param) {
  const auto &f = zone->parface[i_face];
  integer n[3];
  n[0] = blockIdx.x * blockDim.x + threadIdx.x;
  n[1] = blockDim.y * blockIdx.y + threadIdx.y;
  n[2] = blockIdx.z * blockDim.z + threadIdx.z;
  if (n[0] >= f.n_point[0] || n[1] >= f.n_point[1] || n[2] >= f.n_point[2]) return;

  integer idx[3];
  for (int ijk: f.loop_order) {
    idx[ijk] = f.range_start[ijk] + n[ijk] * f.loop_dir[ijk];
  }

  const integer n_var{param->n_scalar + 6}, ngg{zone->ngg};
  integer bias = n_var * (ngg + 1) * (n[f.loop_order[1]] * f.n_point[f.loop_order[2]] + n[f.loop_order[2]]);

  const auto &bv = zone->bv;
#pragma unroll
  for (integer l = 0; l < 6; ++l) {
    data[bias + l] = bv(idx[0], idx[1], idx[2], l);
  }
  const auto& sv=zone->sv;
  for (integer l = 0; l < param->n_scalar; ++l) {
    data[bias + 6 + l] = sv(idx[0], idx[1], idx[2], l);
  }

  for (integer ig = 1; ig <= ngg; ++ig) {
    idx[f.face] -= f.direction;
    bias += n_var;
#pragma unroll
    for (integer l = 0; l < 6; ++l) {
      data[bias + l] = bv(idx[0], idx[1], idx[2], l);
    }
    for (integer l = 0; l < param->n_scalar; ++l) {
      data[bias + 6 + l] = sv(idx[0], idx[1], idx[2], l);
    }
  }
}


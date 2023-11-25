#pragma once

#include "InviscidScheme.cuh"
#include "ViscousScheme.cuh"
#include "Roe.cuh"

namespace cfd {
template<MixtureModel mix_model, TurbulenceMethod turb_method>
void
compute_inviscid_flux(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var,
                      const Parameter &parameter);

template<MixtureModel mix_model, TurbulenceMethod turb_method>
void compute_viscous_flux(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var);

template<MixtureModel mix_model, TurbulenceMethod turb_method>
__global__ void
viscous_flux_fv(cfd::DZone *zone, integer max_extent, cfd::DParameter *param);

template<MixtureModel mix_model, TurbulenceMethod turb_method>
__global__ void
viscous_flux_gv(cfd::DZone *zone, integer max_extent, cfd::DParameter *param);

template<MixtureModel mix_model, TurbulenceMethod turb_method>
__global__ void
viscous_flux_hv(cfd::DZone *zone, integer max_extent, cfd::DParameter *param);

// Implementations

template<MixtureModel mix_model, class turb_method>
void
compute_inviscid_flux(const Block &block, cfd::DZone *zone, DParameter *param, const integer n_var,
                      const Parameter &parameter) {
  const integer inviscid_type=parameter.get_int("inviscid_type");
  switch (inviscid_type) {
    case 0: // Compute the term with primitive reconstruction methods. (MUSCL/NND/1stOrder + LF/AUSM+/HLLC)
      compute_convective_term_pv<mix_model>(block, zone, param, n_var, parameter);
      break;
    case 1: // Compute the term with AWENO methods. (WENO-Z-5 + LF/AUSM+/HLLC)
      compute_convective_term_aweno<mix_model>(block, zone, param, n_var, parameter);
      break;
    case 2: // Roe scheme
    default: // Roe scheme
      Roe_compute_inviscid_flux<mix_model>(block, zone, param, n_var, parameter);
      break;
  }
}

template<MixtureModel mix_model, class turb_method>
void compute_viscous_flux(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var) {
  const integer extent[3]{block.mx, block.my, block.mz};
  const integer dim{extent[2] == 1 ? 2 : 3};
  constexpr integer block_dim = 64;

  dim3 tpb{block_dim, 1, 1};
  dim3 bpg((extent[0] - 1) / (block_dim - 1) + 1, extent[1], extent[2]);
  auto shared_mem = block_dim * n_var * sizeof(real);
  viscous_flux_fv<mix_model, turb_method><<<bpg, tpb, shared_mem>>>(zone, extent[0], param);

  tpb = {1, block_dim, 1};
  bpg = dim3(extent[0], (extent[1] - 1) / (block_dim - 1) + 1, extent[2]);
  viscous_flux_gv<mix_model, turb_method><<<bpg, tpb, shared_mem>>>(zone, extent[1], param);

  if (dim == 3) {
    tpb = {1, 1, block_dim};
    bpg = dim3(extent[0], extent[1], (extent[2] - 1) / (block_dim - 1) + 1);
    viscous_flux_hv<mix_model, turb_method><<<bpg, tpb, shared_mem>>>(zone, extent[2], param);
  }
}

template<MixtureModel mix_model, class turb_method>
__global__ void
viscous_flux_fv(cfd::DZone *zone, integer max_extent, cfd::DParameter *param) {
  integer idx[3];
  idx[0] = ((integer) blockDim.x - 1) * blockIdx.x + threadIdx.x - 1;
  idx[1] = (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  idx[2] = (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (idx[0] >= max_extent) return;
  const auto tid = threadIdx.x;
  const auto n_var{param->n_var};

  extern __shared__ real s[];
  real *fv = s;

  switch (param->viscous_scheme) {
    case 0: // Inviscid computation
      return;
    case 2:
    default: // 2nd order central difference
      compute_fv_2nd_order<mix_model, turb_method>(idx, zone, &fv[tid * n_var], param);
  }
  __syncthreads();

  if (tid > 0) {
    for (integer l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) += fv[tid * n_var + l] - fv[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model, class turb_method>
__global__ void viscous_flux_gv(cfd::DZone *zone, integer max_extent, cfd::DParameter *param) {
  integer idx[3];
  idx[0] = (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  idx[1] = (integer) ((blockDim.y - 1) * blockIdx.y + threadIdx.y) - 1;
  idx[2] = (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (idx[1] >= max_extent) return;
  const auto tid = threadIdx.y;
  const auto n_var{param->n_var};

  extern __shared__ real s[];
  real *gv = s;

  switch (param->viscous_scheme) {
    case 0: // Inviscid computation
      return;
    case 2:
    default: // 2nd order central difference
      compute_gv_2nd_order<mix_model, turb_method>(idx, zone, &gv[tid * n_var], param);
  }
  __syncthreads();

  if (tid > 0) {
    for (integer l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) += gv[tid * n_var + l] - gv[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model, class turb_method>
__global__ void viscous_flux_hv(cfd::DZone *zone, integer max_extent, cfd::DParameter *param) {
  integer idx[3];
  idx[0] = (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  idx[1] = (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  idx[2] = (integer) ((blockDim.z - 1) * blockIdx.z + threadIdx.z) - 1;
  if (idx[2] >= max_extent) return;
  const auto tid = threadIdx.z;
  const auto n_var{param->n_var};

  extern __shared__ real s[];
  real *hv = s;

  switch (param->viscous_scheme) {
    case 0: // Inviscid computation
      return;
    case 2:
    default: // 2nd order central difference
      compute_hv_2nd_order<mix_model, turb_method>(idx, zone, &hv[tid * n_var], param);
  }
  __syncthreads();

  if (tid > 0) {
    for (integer l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) += hv[tid * n_var + l] - hv[(tid - 1) * n_var + l];
    }
  }
}

}
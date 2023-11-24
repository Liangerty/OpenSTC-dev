#include "InviscidScheme.cuh"
#include "LaxFriedrichs.cuh"
#include "AusmP.cuh"
#include "HLLC.cuh"
#include "AWENO.cuh"


namespace cfd {
template<MixtureModel mix_model>
void compute_convective_term_pv(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var,
                                const Parameter &parameter) {
  const integer extent[3]{block.mx, block.my, block.mz};
  constexpr integer block_dim = 64;
  const integer n_computation_per_block = block_dim + 2 * block.ngg - 1;
  auto shared_mem = (block_dim * n_var // fc
                     + n_computation_per_block * (n_var + 3 + 1)) * sizeof(real); // pv[n_var]+metric[3]+jacobian
  if constexpr (mix_model == MixtureModel::FL) {
    // For flamelet model, we need also the mass fractions of species, which is not included in the n_var
    shared_mem += n_computation_per_block * parameter.get_int("n_spec") * sizeof(real);
  }

  for (auto dir = 0; dir < 2; ++dir) {
    integer tpb[3]{1, 1, 1};
    tpb[dir] = block_dim;
    integer bpg[3]{extent[0], extent[1], extent[2]};
    bpg[dir] = (extent[dir] - 1) / (tpb[dir] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    compute_convective_term_pv_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, dir, extent[dir], param);
  }

  if (extent[2] > 1) {
    // 3D computation
    // Number of threads in the 3rd direction cannot exceed 64
    integer tpb[3]{1, 1, 1};
    tpb[2] = 64;
    integer bpg[3]{extent[0], extent[1], extent[2]};
    bpg[2] = (extent[2] - 1) / (tpb[2] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    compute_convective_term_pv_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, 2, extent[2], param);
  }
}

template<MixtureModel mix_model>
__global__ void
compute_convective_term_pv_1D(cfd::DZone *zone, integer direction, integer max_extent, DParameter *param) {
  integer labels[3]{0, 0, 0};
  labels[direction] = 1;
  const integer tid = threadIdx.x * labels[0] + threadIdx.y * labels[1] + threadIdx.z * labels[2];
  const integer block_dim = blockDim.x * blockDim.y * blockDim.z;
  const auto ngg{zone->ngg};
  const integer n_point = block_dim + 2 * ngg - 1;

  integer idx[3];
  idx[0] = (integer) ((blockDim.x - labels[0]) * blockIdx.x + threadIdx.x);
  idx[1] = (integer) ((blockDim.y - labels[1]) * blockIdx.y + threadIdx.y);
  idx[2] = (integer) ((blockDim.z - labels[2]) * blockIdx.z + threadIdx.z);
  idx[direction] -= 1;
  if (idx[direction] >= max_extent) return;

  // load variables to shared memory
  extern __shared__ real s[];
  const auto n_var{param->n_var};
  auto n_reconstruct{n_var};
  if constexpr (mix_model == MixtureModel::FL) {
    n_reconstruct += param->n_spec;
  }
  real *pv = s;
  real *metric = &pv[n_point * n_reconstruct];
  real *jac = &metric[n_point * 3];
  real *fc = &jac[n_point];

  const integer i_shared = tid - 1 + ngg;
  for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
    pv[i_shared * n_reconstruct + l] = zone->bv(idx[0], idx[1], idx[2], l);
  }
  const auto n_scalar{param->n_scalar};
  for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
    pv[i_shared * n_reconstruct + 5 + l] = zone->sv(idx[0], idx[1], idx[2], l);
  }
  for (auto l = 1; l < 4; ++l) {
    metric[i_shared * 3 + l - 1] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, l);
  }
  jac[i_shared] = zone->jac(idx[0], idx[1], idx[2]);

  // ghost cells
  if (tid == 0) {
    // Responsible for the left (ngg-1) points
    for (auto i = 1; i < ngg; ++i) {
      const auto ig_shared = ngg - 1 - i;
      const integer g_idx[3]{idx[0] - i * labels[0], idx[1] - i * labels[1], idx[2] - i * labels[2]};

      for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
        pv[ig_shared * n_reconstruct + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, Z, Z_prime...
        pv[ig_shared * n_reconstruct + 5 + l] = zone->sv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  if (tid == block_dim - 1 || idx[direction] == max_extent - 1) {
    // Responsible for the right ngg points
    for (auto i = 1; i <= ngg; ++i) {
      const auto ig_shared = tid + i + ngg - 1;
      const integer g_idx[3]{idx[0] + i * labels[0], idx[1] + i * labels[1], idx[2] + i * labels[2]};

      for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
        pv[ig_shared * n_reconstruct + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, Z, Z_prime...
        pv[ig_shared * n_reconstruct + 5 + l] = zone->sv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  __syncthreads();

  // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
  constexpr integer n_reconstruction_max =
      7 + MAX_SPEC_NUMBER + 4 + 2; // rho,u,v,w,p,Y_{1...Ns},(k,omega,z,z_prime),E,gamma
  real pv_l[n_reconstruction_max], pv_r[n_reconstruction_max];
  reconstruction<mix_model>(pv, pv_l, pv_r, i_shared, param);
  __syncthreads();

  // compute the half-point numerical flux with the chosen Riemann solver
  switch (param->inviscid_scheme) {
    case 1:
      compute_lf_flux<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    case 3: // AUSM+
      compute_ausmPlus_flux<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    case 4: // HLLC
      compute_hllc_flux<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    default:
      compute_ausmPlus_flux<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
  }
  __syncthreads();

  if (tid > 0) {
    for (integer l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model>
void compute_convective_term_aweno(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var,
                                   const Parameter &parameter) {
  // The implementation of AWENO is based on Fig.9 of (Ye, C-C, Zhang, P-J-Y, Wan, Z-H, and Sun, D-J (2022)
  // An alternative formulation of targeted ENO scheme for hyperbolic conservation laws. Computers & Fluids, 238, 105368.
  // doi:10.1016/j.compfluid.2022.105368.)

  const integer extent[3]{block.mx, block.my, block.mz};

  constexpr integer block_dim = 64;
  const integer n_computation_per_block = block_dim + 2 * block.ngg - 1;
  auto shared_mem = (block_dim * n_var // fc
                     + n_computation_per_block * (n_var + 3)) * sizeof(real) // cv[n_var]+p+T+jacobian
                    + n_computation_per_block * 3 * sizeof(real); // metric[3]
  auto shared_cds = block_dim * n_var * sizeof(real); // f_i

  for (auto dir = 0; dir < 2; ++dir) {
    integer tpb[3]{1, 1, 1};
    tpb[dir] = block_dim;
    integer bpg[3]{extent[0], extent[1], extent[2]};
    bpg[dir] = (extent[dir] - 1) / (tpb[dir] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    compute_convective_term_aweno_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, dir, extent[dir], param);

    bpg[dir] = (extent[dir] - 1) / (tpb[dir] - 2 * block.ngg) + 1;

    dim3 BPG2(bpg[0], bpg[1], bpg[2]);
    CDSPart1D<mix_model><<<BPG, TPB, shared_cds>>>(zone, dir, extent[dir], param);
  }

  if (extent[2] > 1) {
    // 3D computation
    // Number of threads in the 3rd direction cannot exceed 64
    integer tpb[3]{1, 1, 1};
    tpb[2] = 64;
    integer bpg[3]{extent[0], extent[1], extent[2]};
    bpg[2] = (extent[2] - 1) / (tpb[2] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    compute_convective_term_aweno_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, 2, extent[2], param);

    bpg[2] = (extent[2] - 1) / (tpb[2] - 2 * block.ngg) + 1;


    dim3 BPG2(bpg[0], bpg[1], bpg[2]);
    CDSPart1D<mix_model><<<BPG, TPB, shared_cds>>>(zone, 2, extent[2], param);
  }
}

template<MixtureModel mix_model>
__global__ void
compute_convective_term_aweno_1D(cfd::DZone *zone, integer direction, integer max_extent, DParameter *param) {
  integer labels[3]{0, 0, 0};
  labels[direction] = 1;
  const auto tid = (integer) (threadIdx.x * labels[0] + threadIdx.y * labels[1] + threadIdx.z * labels[2]);
  const auto block_dim = (integer) (blockDim.x * blockDim.y * blockDim.z);
  const auto ngg{zone->ngg};
  const integer n_point = block_dim + 2 * ngg - 1;

  integer idx[3];
  idx[0] = (integer) ((blockDim.x - labels[0]) * blockIdx.x + threadIdx.x);
  idx[1] = (integer) ((blockDim.y - labels[1]) * blockIdx.y + threadIdx.y);
  idx[2] = (integer) ((blockDim.z - labels[2]) * blockIdx.z + threadIdx.z);
  idx[direction] -= 1;
  if (idx[direction] >= max_extent) return;

  // load variables to shared memory
  extern __shared__ real s[];
  const auto n_var{param->n_var};
  auto n_reconstruct{n_var + 2};
  real *cv = s;
  real *metric = &cv[n_point * n_reconstruct];
  real *jac = &metric[n_point * 3];
  real *fc = &jac[n_point];


  const integer i_shared = tid - 1 + ngg;
  for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
    cv[i_shared * n_reconstruct + l] = zone->cv(idx[0], idx[1], idx[2], l);
  }
  cv[i_shared * n_reconstruct + n_var] = zone->bv(idx[0], idx[1], idx[2], 4);
  cv[i_shared * n_reconstruct + n_var + 1] = zone->bv(idx[0], idx[1], idx[2], 5);
  for (auto l = 1; l < 4; ++l) {
    metric[i_shared * 3 + l - 1] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, l);
  }
  jac[i_shared] = zone->jac(idx[0], idx[1], idx[2]);

  // ghost cells
  if (tid == 0) {
    // Responsible for the left (ngg-1) points
    for (auto i = 1; i < ngg; ++i) {
      const auto ig_shared = ngg - 1 - i;
      const integer g_idx[3]{idx[0] - i * labels[0], idx[1] - i * labels[1], idx[2] - i * labels[2]};

      for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
        cv[ig_shared * n_reconstruct + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      cv[ig_shared * n_reconstruct + n_var] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
      cv[ig_shared * n_reconstruct + n_var + 1] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 5);
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  if (tid == block_dim - 1 || idx[direction] == max_extent - 1) {
    // Responsible for the right ngg points
    for (auto i = 1; i <= ngg; ++i) {
      const auto ig_shared = tid + i + ngg - 1;
      const integer g_idx[3]{idx[0] + i * labels[0], idx[1] + i * labels[1], idx[2] + i * labels[2]};

      for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
        cv[ig_shared * n_reconstruct + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      cv[ig_shared * n_reconstruct + n_var] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
      cv[ig_shared * n_reconstruct + n_var + 1] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 5);
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  __syncthreads();

  // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
  constexpr integer n_reconstruction_max =
      7 + MAX_SPEC_NUMBER + 4 + 2; // rho,u,v,w,p,Y_{1...Ns},(k,omega,z,z_prime),E,gamma
  real pv_l[n_reconstruction_max], pv_r[n_reconstruction_max];
  AWENO_interpolation<mix_model>(cv, pv_l, pv_r, i_shared, n_var, metric, param);
  __syncthreads();

  // compute the half-point numerical flux with the chosen Riemann solver
  switch (param->inviscid_scheme) {
    case 1:
      compute_lf_flux<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    case 3: // AUSM+
      compute_ausmPlus_flux<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    case 4: // HLLC
      compute_hllc_flux<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    default:
      compute_ausmPlus_flux<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
  }
  __syncthreads();

  if (tid > 0) {
    for (integer l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

// template instantiation
template void
compute_convective_term_pv<MixtureModel::Air>(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var,
                                              const Parameter &parameter);
template void
compute_convective_term_pv<MixtureModel::Mixture>(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var,
                                                  const Parameter &parameter);
template void
compute_convective_term_pv<MixtureModel::MixtureFraction>(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var,
                                                          const Parameter &parameter);
template void
compute_convective_term_pv<MixtureModel::FR>(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var,
                                             const Parameter &parameter);
template void
compute_convective_term_pv<MixtureModel::FL>(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var,
                                             const Parameter &parameter);
template void
compute_convective_term_aweno<MixtureModel::Air>(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var,
                                              const Parameter &parameter);
template void
compute_convective_term_aweno<MixtureModel::Mixture>(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var,
                                                  const Parameter &parameter);
template void
compute_convective_term_aweno<MixtureModel::MixtureFraction>(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var,
                                                          const Parameter &parameter);
template void
compute_convective_term_aweno<MixtureModel::FR>(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var,
                                             const Parameter &parameter);
template void
compute_convective_term_aweno<MixtureModel::FL>(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var,
                                             const Parameter &parameter);
}
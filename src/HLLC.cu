#include "HLLC.cuh"
#include "Parameter.h"
#include "Field.h"
#include "DParameter.cuh"
#include "Constants.h"
#include "Thermo.cuh"
#include "Parallel.h"
#include "Reconstruction.cuh"

namespace cfd {

template<MixtureModel mix_model>
void HLLC_compute_inviscid_flux(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var,
                                const Parameter &parameter) {
  const integer extent[3]{block.mx, block.my, block.mz};

  constexpr integer block_dim = 128;
  const integer n_computation_per_block = block_dim + 2 * block.ngg - 1;
  auto shared_mem = (block_dim * n_var // fc
                     + n_computation_per_block * (n_var + 1)) * sizeof(real) // pv[n_var]+jacobian
                    + n_computation_per_block * 3 * sizeof(real); // metric[3]
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
    HLLC_compute_inviscid_flux_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, dir, extent[dir], param);
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
    HLLC_compute_inviscid_flux_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, 2, extent[2], param);
  }
}

template<MixtureModel mix_model>
__global__ void
HLLC_compute_inviscid_flux_1D(cfd::DZone *zone, integer direction, integer max_extent, DParameter *param) {
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

  HLLC_compute_half_point_flux<mix_model>(zone, pv, tid, param, fc, metric, jac);
  __syncthreads();

  if (tid > 0) {
    for (integer l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model>
__device__ void
HLLC_compute_half_point_flux(DZone *zone, real *pv, integer tid, DParameter *param, real *fc, const real *metric,
                             const real *jac) {
  constexpr integer n_reconstruction_max =
      7 + MAX_SPEC_NUMBER + 4; // rho,u,v,w,p,Y_{1...Ns},(k,omega,z,z_prime),E,gamma
  real pv_l[n_reconstruction_max], pv_r[n_reconstruction_max];
  const integer i_shared = tid - 1 + zone->ngg;
  reconstruction<mix_model>(pv, pv_l, pv_r, i_shared, zone, param);

  const integer n_var = param->n_var;
  integer n_reconstruct{n_var};
  if constexpr (mix_model == MixtureModel::FL) {
    n_reconstruct += param->n_spec;
  }

  // Compute the Roe averaged variables.
  const real rl_c{std::sqrt(pv_l[0]) / (std::sqrt(pv_l[0]) + std::sqrt(pv_r[0]))}, rr_c{
      std::sqrt(pv_r[0]) / (std::sqrt(pv_l[0]) + std::sqrt(pv_r[0]))};
  const real u_tilde{pv_l[1] * rl_c + pv_r[1] * rr_c};
  const real v_tilde{pv_l[2] * rl_c + pv_r[2] * rr_c};
  const real w_tilde{pv_l[3] * rl_c + pv_r[3] * rr_c};
  const real hl{(pv_l[n_reconstruct] + pv_l[4]) / pv_l[0]};
  const real hr{(pv_r[n_reconstruct] + pv_r[4]) / pv_r[0]};
  const real h_tilde{hl * rl_c + hr * rr_c};
  const real ek_tilde{0.5 * (u_tilde * u_tilde + v_tilde * v_tilde + w_tilde * w_tilde)};

  real gamma{gamma_air};
  real c_tilde{0};
  real svm[MAX_SPEC_NUMBER + 4];
  memset(svm, 0, sizeof(real) * (MAX_SPEC_NUMBER + 4));
  for (integer l = 0; l < param->n_var - 5; ++l) {
    svm[l] = rl_c * pv_l[l + 5] + rr_c * pv_r[l + 5];
  }
  real h_i[MAX_SPEC_NUMBER];

  if constexpr (mix_model == MixtureModel::Air) {
    c_tilde = std::sqrt((gamma - 1) * (h_tilde - ek_tilde));
  } else {
    real mw_inv{0};
    for (integer l = 0; l < param->n_spec; ++l) {
      mw_inv += svm[l] / param->mw[l];
    }

    const real tl{pv_l[4] / pv_l[0]};
    const real tr{pv_r[4] / pv_r[0]};
    const real t{(rl_c * tl + rr_c * tr) / (R_u * mw_inv)};

    real cp_i[MAX_SPEC_NUMBER];
    compute_enthalpy_and_cp(t, h_i, cp_i, param);
    real cv{0}, cp{0};
    for (integer l = 0; l < param->n_spec; ++l) {
      cp += cp_i[l] * svm[l];
      cv += svm[l] * (cp_i[l] - R_u / param->mw[l]);
    }
    gamma = cp / cv;
    c_tilde = std::sqrt(gamma * R_u * mw_inv * t);
  }

  real kx = 0.5 * (metric[i_shared * 3] + metric[(i_shared + 1) * 3]);
  real ky = 0.5 * (metric[i_shared * 3 + 1] + metric[(i_shared + 1) * 3 + 1]);
  real kz = 0.5 * (metric[i_shared * 3 + 2] + metric[(i_shared + 1) * 3 + 2]);
  const real gradK = std::sqrt(kx * kx + ky * ky + kz * kz);
  const real U_tilde_bar{(kx * u_tilde + ky * v_tilde + kz * w_tilde) / gradK};

  auto fci = &fc[tid * n_var];
  const real jac_ave{0.5 * (jac[i_shared] + jac[i_shared + 1])};

  const real Ul{kx * pv_l[1] + ky * pv_l[2] + kz * pv_l[3]};
  const real cl{std::sqrt(pv_l[n_reconstruct + 1] * pv_l[4] / pv_l[0])};
  const real sl{min(Ul / gradK - cl, U_tilde_bar - c_tilde)};
  if (sl >= 0) {
    // The flow is supersonic from left to right, the flux is computed from the left value.
    const real rhoUk{pv_l[0] * Ul};
    fci[0] = jac_ave * rhoUk;
    fci[1] = jac_ave * (rhoUk * pv_l[1] + pv_l[4] * kx);
    fci[2] = jac_ave * (rhoUk * pv_l[2] + pv_l[4] * ky);
    fci[3] = jac_ave * (rhoUk * pv_l[3] + pv_l[4] * kz);
    fci[4] = jac_ave * ((pv_l[n_reconstruct] + pv_l[4]) * Ul);
    for (integer l = 5; l < n_var; ++l) {
      fci[l] = jac_ave * rhoUk * pv_l[l];
    }
    return;
  }

  const real Ur{kx * pv_r[1] + ky * pv_r[2] + kz * pv_r[3]};
  const real cr{std::sqrt(pv_r[n_reconstruct + 1] * pv_r[4] / pv_r[0])};
  const real sr{max(Ur / gradK + cr, U_tilde_bar + c_tilde)};
  if (sr < 0) {
    // The flow is supersonic from right to left, the flux is computed from the right value.
    const real rhoUk{pv_r[0] * Ur};
    fci[0] = jac_ave * rhoUk;
    fci[1] = jac_ave * (rhoUk * pv_r[1] + pv_r[4] * kx);
    fci[2] = jac_ave * (rhoUk * pv_r[2] + pv_r[4] * ky);
    fci[3] = jac_ave * (rhoUk * pv_r[3] + pv_r[4] * kz);
    fci[4] = jac_ave * ((pv_r[n_reconstruct] + pv_r[4]) * Ur);
    for (integer l = 5; l < n_var; ++l) {
      fci[l] = jac_ave * rhoUk * pv_r[l];
    }
    return;
  }

  // Else, the current position is in star region; we need to identify the left and right star states.
  const real sm{((pv_r[0] * Ur * (sr - Ur / gradK) - pv_l[0] * Ul * (sl - Ul / gradK)) / gradK + pv_l[4] - pv_r[4]) /
                (pv_r[0] * (sr - Ur / gradK) - pv_l[0] * (sl - Ul / gradK))};
  const real pm{pv_l[0] * (sl - Ul / gradK) * (sm - Ul / gradK) + pv_l[4]};
  if (sm >= 0) {
    // Left star region, F_{*L}
    const real pCoeff{jac_ave / (sl - sm)};
    const real QlCoeff{jac_ave * pCoeff * sm * (sl * gradK - Ul) * pv_l[0]};
    fci[0] = QlCoeff;
    const real dP{(sl * pm - sm * pv_l[4]) * pCoeff};
    fci[1] = QlCoeff * pv_l[1] + dP * kx;
    fci[2] = QlCoeff * pv_l[2] + dP * ky;
    fci[3] = QlCoeff * pv_l[3] + dP * kz;
    fci[4] = QlCoeff * pv_l[n_reconstruct] / pv_l[0] + pCoeff * (sl * pm * sm * gradK - sm * pv_l[4] * Ul);
    for (integer l = 5; l < n_var; ++l) {
      fci[l] = QlCoeff * pv_l[l];
    }
  } else {
    // Right star region, F_{*R}
    const real pCoeff{jac_ave / (sr - sm)};
    const real QrCoeff{jac_ave * pCoeff * sm * (sr * gradK - Ur) * pv_r[0]};
    fci[0] = QrCoeff;
    const real dP{(sr * pm - sm * pv_r[4]) * pCoeff};
    fci[1] = QrCoeff * pv_r[1] + dP * kx;
    fci[2] = QrCoeff * pv_r[2] + dP * ky;
    fci[3] = QrCoeff * pv_r[3] + dP * kz;
    fci[4] = QrCoeff * pv_r[n_reconstruct] / pv_r[0] + pCoeff * (sr * pm * sm * gradK - sm * pv_r[4] * Ur);
    for (integer l = 5; l < n_var; ++l) {
      fci[l] = QrCoeff * pv_r[l];
    }
  }
}

template
void HLLC_compute_inviscid_flux<MixtureModel::Air>(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var, const Parameter &parameter);
template
void HLLC_compute_inviscid_flux<MixtureModel::Mixture>(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var, const Parameter &parameter);
template
void HLLC_compute_inviscid_flux<MixtureModel::FR>(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var, const Parameter &parameter);
template<>
void HLLC_compute_inviscid_flux<MixtureModel::FL>(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var, const Parameter &parameter) {
  printf("HLLC_compute_inviscid_flux<MixtureModel::FL> is not implemented yet.\n");
  MpiParallel::exit();
}
template<>
void HLLC_compute_inviscid_flux<MixtureModel::MixtureFraction>(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var, const Parameter &parameter) {
  printf("HLLC_compute_inviscid_flux<MixtureModel::MixtureFraction> is not implemented yet.\n");
  MpiParallel::exit();
}
}

#include "Roe.cuh"
#include "DParameter.cuh"
#include "Field.h"
#include "Reconstruction.cuh"
#include "Thermo.cuh"
#include "Parallel.h"

namespace cfd {
template<MixtureModel mix_model>
void Roe_compute_inviscid_flux(const Block &block, cfd::DZone *zone, DParameter *param, const integer n_var,
                               const Parameter &parameter) {
  const integer extent[3]{block.mx, block.my, block.mz};

  // Compute the entropy fix delta
  dim3 thread_per_block{8, 8, 4};
  if (extent[2] == 1) {
    thread_per_block = {16, 16, 1};
  }
  dim3 block_per_grid{(extent[0] + 1) / thread_per_block.x + 1,
                      (extent[1] + 1) / thread_per_block.y + 1,
                      (extent[2] + 1) / thread_per_block.z + 1};
  compute_entropy_fix_delta<mix_model><<<block_per_grid, thread_per_block>>>(zone, param);

  constexpr integer block_dim = 128;
  const integer n_computation_per_block = block_dim + 2 * block.ngg - 1;
  auto shared_mem = (block_dim * n_var // fc
                     + n_computation_per_block * (n_var + 1)) * sizeof(real) // pv[n_var]+jacobian
                    + n_computation_per_block * 3 * sizeof(real) // metric[3]
                    + n_computation_per_block * sizeof(real); // entropy fix delta
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
    Roe_compute_inviscid_flux_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, dir, extent[dir], param);
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
    Roe_compute_inviscid_flux_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, 2, extent[2], param);
  }
}

template<MixtureModel mix_model>
__global__ void compute_entropy_fix_delta(cfd::DZone *zone, DParameter *param) {
  const integer mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = (integer)(blockDim.x * blockIdx.x + threadIdx.x) - 1;
  integer j = (integer)(blockDim.y * blockIdx.y + threadIdx.y) - 1;
  integer k = (integer)(blockDim.z * blockIdx.z + threadIdx.z) - 1;
  if (i >= mx + 1 || j >= my + 1 || k >= mz + 1) return;

  const auto &bv{zone->bv};
  const auto &metric{zone->metric(i, j, k)};

  const real U = abs(bv(i, j, k, 1) * metric(1, 1) + bv(i, j, k, 2) * metric(1, 2) + bv(i, j, k, 3) * metric(1, 3));
  const real V = abs(bv(i, j, k, 1) * metric(2, 1) + bv(i, j, k, 2) * metric(2, 2) + bv(i, j, k, 3) * metric(2, 3));
  const real W = abs(bv(i, j, k, 1) * metric(3, 1) + bv(i, j, k, 2) * metric(3, 2) + bv(i, j, k, 3) * metric(3, 3));

  const real kx = sqrt(metric(1, 1) * metric(1, 1) + metric(1, 2) * metric(1, 2) + metric(1, 3) * metric(1, 3));
  const real ky = sqrt(metric(2, 1) * metric(2, 1) + metric(2, 2) * metric(2, 2) + metric(2, 3) * metric(2, 3));
  const real kz = sqrt(metric(3, 1) * metric(3, 1) + metric(3, 2) * metric(3, 2) + metric(3, 3) * metric(3, 3));

  if (param->dim == 2) {
    zone->entropy_fix_delta(i, j, k) =
        param->entropy_fix_factor * (U + V + zone->acoustic_speed(i, j, k) * 0.5 * (kx + ky));
  } else {
    // 3D
    zone->entropy_fix_delta(i, j, k) =
        param->entropy_fix_factor * (U + V + W + zone->acoustic_speed(i, j, k) * (kx + ky + kz) / 3.0);
  }
}

template<MixtureModel mix_model>
__global__ void
Roe_compute_inviscid_flux_1D(cfd::DZone *zone, integer direction, integer max_extent, DParameter *param) {
  integer labels[3]{0, 0, 0};
  labels[direction] = 1;
  const auto tid = (integer)(threadIdx.x * labels[0] + threadIdx.y * labels[1] + threadIdx.z * labels[2]);
  const auto block_dim = (integer)(blockDim.x * blockDim.y * blockDim.z);
  const auto ngg{zone->ngg};
  const integer n_point = block_dim + 2 * ngg - 1;

  integer idx[3];
  idx[0] = (integer)((blockDim.x - labels[0]) * blockIdx.x + threadIdx.x);
  idx[1] = (integer)((blockDim.y - labels[1]) * blockIdx.y + threadIdx.y);
  idx[2] = (integer)((blockDim.z - labels[2]) * blockIdx.z + threadIdx.z);
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
  real *entropy_fix_delta = &jac[n_point];
  real *fc = &entropy_fix_delta[n_point];

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
  entropy_fix_delta[i_shared] = zone->entropy_fix_delta(idx[0], idx[1], idx[2]);

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
    entropy_fix_delta[tid + ngg] = zone->entropy_fix_delta(idx[0] + labels[0], idx[1] + labels[1], idx[2] + labels[2]);
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

  Roe_compute_half_point_flux<mix_model>(zone, pv, tid, param, fc, metric, jac, entropy_fix_delta,
                                         direction);
  __syncthreads();


  if (tid > 0) {
    for (integer l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}


template<MixtureModel mix_model>
__device__ void
Roe_compute_half_point_flux(DZone *zone, real *pv, integer tid, DParameter *param, real *fc, real *metric,
                            const real *jac, const real *entropy_fix_delta, integer direction) {
  constexpr integer n_reconstruction_max =
      7 + MAX_SPEC_NUMBER + 4; // rho,u,v,w,p,Y_{1...Ns},(k,omega,z,z_prime),E,gamma
  real pv_l[n_reconstruction_max], pv_r[n_reconstruction_max];
  const integer i_shared = tid - 1 + zone->ngg;
  reconstruction<mix_model>(pv, pv_l, pv_r, i_shared, param);

  // The entropy fix delta may not need shared memory, which may be replaced by shuffle instructions.
  integer n_reconstruct{param->n_var};
  if constexpr (mix_model == MixtureModel::FL) {
    n_reconstruct += param->n_spec;
  }

  // Compute the left and right convective fluxes, which uses the reconstructed primitive variables, rather than the roe averaged ones.
  auto fci = &fc[tid * param->n_var];
  compute_half_sum_left_right_flux<mix_model>(pv_l, pv_r, param, jac, metric, i_shared, direction, fci);

  // Compute the Roe averaged variables.
  const real dl = std::sqrt(pv_l[0]), dr = std::sqrt(pv_r[0]);
  const real inv_denominator = 1.0 / (dl + dr);
  const real u = (dl * pv_l[1] + dr * pv_r[1]) * inv_denominator;
  const real v = (dl * pv_l[2] + dr * pv_r[2]) * inv_denominator;
  const real w = (dl * pv_l[3] + dr * pv_r[3]) * inv_denominator;
  const real ek = 0.5 * (u * u + v * v + w * w);
  const real hl = (pv_l[n_reconstruct] + pv_l[4]) / pv_l[0];
  const real hr = (pv_r[n_reconstruct] + pv_r[4]) / pv_r[0];
  const real h = (dl * hl + dr * hr) * inv_denominator;

  real gamma{gamma_air};
  real c = std::sqrt((gamma - 1) * (h - ek));
  real mw{mw_air};
  real svm[MAX_SPEC_NUMBER + 4];
  memset(svm, 0, sizeof(real) * (MAX_SPEC_NUMBER + 4));
  for (integer l = 0; l < param->n_var - 5; ++l) {
    svm[l] = (dl * pv_l[l + 5] + dr * pv_r[l + 5]) * inv_denominator;
  }

  real h_i[MAX_SPEC_NUMBER];
  if constexpr (mix_model != MixtureModel::Air) {
    real mw_inv{0};
    for (integer l = 0; l < param->n_spec; ++l) {
      mw_inv += svm[l] / param->mw[l];
    }

    const real tl{pv_l[4] / pv_l[0]};
    const real tr{pv_r[4] / pv_r[0]};
    const real t{(dl * tl + dr * tr) * inv_denominator / (R_u * mw_inv)};

    real cp_i[MAX_SPEC_NUMBER];
    compute_enthalpy_and_cp(t, h_i, cp_i, param);
    real cv{0}, cp{0};
    for (integer l = 0; l < param->n_spec; ++l) {
      cp += cp_i[l] * svm[l];
      cv += svm[l] * (cp_i[l] - R_u / param->mw[l]);
    }
    gamma = cp / cv;
    c = std::sqrt(gamma * R_u * mw_inv * t);
    mw = 1.0 / mw_inv;
  }

  // Compute the characteristics
  real kx = 0.5 * (metric[i_shared * 3] + metric[(i_shared + 1) * 3]);
  real ky = 0.5 * (metric[i_shared * 3 + 1] + metric[(i_shared + 1) * 3 + 1]);
  real kz = 0.5 * (metric[i_shared * 3 + 2] + metric[(i_shared + 1) * 3 + 2]);
  const real gradK = std::sqrt(kx * kx + ky * ky + kz * kz);
  real Uk = kx * u + ky * v + kz * w;

  real characteristic[3]{Uk - gradK * c, Uk, Uk + gradK * c};
  // entropy fix
  const real entropy_fix_delta_ave{0.5 * (entropy_fix_delta[i_shared] + entropy_fix_delta[i_shared + 1])};
  for (auto &cc: characteristic) {
    cc = std::abs(cc);
    if (cc < entropy_fix_delta_ave) {
      cc = 0.5 * (cc * cc / entropy_fix_delta_ave + entropy_fix_delta_ave);
    }
  }

  kx /= gradK;
  ky /= gradK;
  kz /= gradK;
  Uk /= gradK;

  // compute dQ
  const real jac_ave{0.5 * (jac[i_shared] + jac[i_shared + 1])};
  real dq[5 + MAX_SPEC_NUMBER + 4];
  memset(dq, 0, sizeof(real) * (5 + MAX_SPEC_NUMBER + 4));
  dq[0] = jac_ave * (pv_r[0] - pv_l[0]);
  for (integer l = 1; l < param->n_var; ++l) {
    dq[l] = jac_ave * (pv_r[0] * pv_r[l] - pv_l[0] * pv_l[l]);
  }
  dq[4] = jac_ave * (pv_r[n_reconstruct] - pv_l[n_reconstruct]);

  real c1 = (gamma - 1) * (ek * dq[0] - u * dq[1] - v * dq[2] - w * dq[3] + dq[4]) / (c * c);
  real c2 = (kx * dq[1] + ky * dq[2] + kz * dq[3] - Uk * dq[0]) / c;
  for (integer l = 0; l < param->n_spec; ++l) {
    c1 += (mw / param->mw[l] - h_i[l] * (gamma - 1) / (c * c)) * dq[l + 5];
  }
  real c3 = dq[0] - c1;

  // compute L*dQ
  real LDq[5 + MAX_SPEC_NUMBER + 4];
  memset(LDq, 0, sizeof(real) * (5 + MAX_SPEC_NUMBER + 4));
  LDq[0] = 0.5 * (c1 - c2);
  LDq[1] = kx * c3 - ((kz * v - ky * w) * dq[0] - kz * dq[2] + ky * dq[3]) / c;
  LDq[2] = ky * c3 - ((kx * w - kz * u) * dq[0] - kx * dq[3] + kz * dq[1]) / c;
  LDq[3] = kz * c3 - ((ky * u - kx * v) * dq[0] - ky * dq[1] + kx * dq[2]) / c;
  LDq[4] = 0.5 * (c1 + c2);
  for (integer l = 0; l < param->n_scalar_transported; ++l) {
    if constexpr (mix_model != MixtureModel::FL)
      LDq[l + 5] = dq[l + 5] - svm[l] * dq[0];
    else
      LDq[l + 5] = dq[l + 5] - svm[l + param->n_spec] * dq[0];
  }

  // To reduce memory usage, we use dq array to contain the b array to be computed
  auto b = dq;
  b[0] = -characteristic[0] * LDq[0];
  for (integer l = 1; l < param->n_var; ++l) {
    b[l] = -characteristic[1] * LDq[l];
  }
  b[4] = -characteristic[2] * LDq[4];

  const real c0 = kx * b[1] + ky * b[2] + kz * b[3];
  c1 = c0 + b[0] + b[4];
  c2 = c * (b[4] - b[0]);
  c3 = 0;
  for (integer l = 0; l < param->n_spec; ++l)
    c3 += (h_i[l] - mw / param->mw[l] * c * c / (gamma - 1)) * b[l + 5];

  fci[0] += 0.5 * c1;
  fci[1] += 0.5 * (u * c1 + kx * c2 - c * (kz * b[2] - ky * b[3]));
  fci[2] += 0.5 * (v * c1 + ky * c2 - c * (kx * b[3] - kz * b[1]));
  fci[3] += 0.5 * (w * c1 + kz * c2 - c * (ky * b[1] - kx * b[2]));
  fci[4] += 0.5 *
            (h * c1 + Uk * c2 - c * c * c0 / (gamma - 1) + c * ((kz * v - ky * w) * b[1] + (kx * w - kz * u) * b[2] +
                                                                (ky * u - kx * v) * b[3]) + c3);
  for (integer l = 0; l < param->n_var - 5; ++l)
    fci[5 + l] += 0.5 * (b[l + 5] + svm[l] * c1);
}

template<MixtureModel mixtureModel>
__device__ void
compute_half_sum_left_right_flux(const real *pv_l, const real *pv_r, DParameter *param, const real *jac,
                                 const real *metric,
                                 integer i_shared, integer direction, real *fc) {
  real JacKx = jac[i_shared] * metric[i_shared * 3];
  real JacKy = jac[i_shared] * metric[i_shared * 3 + 1];
  real JacKz = jac[i_shared] * metric[i_shared * 3 + 2];
  real Uk = pv_l[1] * JacKx + pv_l[2] * JacKy + pv_l[3] * JacKz;

  integer n_reconstruct{param->n_var};
  if constexpr (mixtureModel == MixtureModel::FL) {
    n_reconstruct += param->n_spec;
  }
  real coeff = Uk * pv_l[0];
  fc[0] = 0.5 * coeff;
  fc[1] = 0.5 * (coeff * pv_l[1] + pv_l[4] * JacKx);
  fc[2] = 0.5 * (coeff * pv_l[2] + pv_l[4] * JacKy);
  fc[3] = 0.5 * (coeff * pv_l[3] + pv_l[4] * JacKz);
  fc[4] = 0.5 * Uk * (pv_l[4] + pv_l[n_reconstruct]);
  for (integer l = 5; l < param->n_var; ++l) {
    if constexpr (mixtureModel != MixtureModel::FL) {
      fc[l] = 0.5 * coeff * pv_l[l];
    } else {
      fc[l] = 0.5 * coeff * pv_l[l + param->n_spec];
    }
  }

  JacKx = jac[i_shared + 1] * metric[(i_shared + 1) * 3];
  JacKy = jac[i_shared + 1] * metric[(i_shared + 1) * 3 + 1];
  JacKz = jac[i_shared + 1] * metric[(i_shared + 1) * 3 + 2];
  Uk = pv_r[1] * JacKx + pv_r[2] * JacKy + pv_r[3] * JacKz;

  coeff = Uk * pv_r[0];
  fc[0] += 0.5 * coeff;
  fc[1] += 0.5 * (coeff * pv_r[1] + pv_r[4] * JacKx);
  fc[2] += 0.5 * (coeff * pv_r[2] + pv_r[4] * JacKy);
  fc[3] += 0.5 * (coeff * pv_r[3] + pv_r[4] * JacKz);
  fc[4] += 0.5 * Uk * (pv_r[4] + pv_r[n_reconstruct]);
  for (integer l = 5; l < param->n_var; ++l) {
    if constexpr (mixtureModel != MixtureModel::FL) {
      fc[l] += 0.5 * coeff * pv_r[l];
    } else {
      fc[l] += 0.5 * coeff * pv_r[l + param->n_spec];
    }
  }
}

template
void Roe_compute_inviscid_flux<MixtureModel::Air>(const Block &block, cfd::DZone *zone, DParameter *param,
                                                  const integer n_var, const Parameter &parameter);

template
void Roe_compute_inviscid_flux<MixtureModel::Mixture>(const Block &block, cfd::DZone *zone, DParameter *param,
                                                      const integer n_var, const Parameter &parameter);

template
void Roe_compute_inviscid_flux<MixtureModel::FR>(const Block &block, cfd::DZone *zone, DParameter *param,
                                                 const integer n_var, const Parameter &parameter);

template<>
void Roe_compute_inviscid_flux<MixtureModel::FL>(const Block &block, cfd::DZone *zone, DParameter *param,
                                                 const integer n_var, const Parameter &parameter) {
  printf("Roe_compute_inviscid_flux<MixtureModel::FL> is not implemented yet.\n");
  MpiParallel::exit();
}

template<>
void Roe_compute_inviscid_flux<MixtureModel::MixtureFraction>(const Block &block, cfd::DZone *zone, DParameter *param,
                                                              const integer n_var, const Parameter &parameter) {
  printf("Roe_compute_inviscid_flux<MixtureModel::MixtureFraction> is not implemented yet.\n");
  MpiParallel::exit();
}
}
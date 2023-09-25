#pragma once

#include "Define.h"
#include "Field.h"
#include "DParameter.cuh"
#include "Reconstruction.cuh"
#include "Thermo.cuh"

namespace cfd {
template<MixtureModel mix_model>
__device__ void
compute_entropy_fix_delta(const real *pv, const real *metric, DParameter *param, real *entropy_fix_delta, integer tid);

template<MixtureModel mixtureModel>
__device__ void
compute_half_sum_left_right_flux(real *pv_l, real *pv_r, DParameter *param, const real *jac, real *metric,
                                 integer i_shared, integer direction, real *fc);

template<MixtureModel mix_model, class turb_method>
__device__ void
Roe_compute_inviscid_flux(DZone *zone, real *pv, integer tid, DParameter *param, real *fc, real *metric,
                          const real *jac, real *entropy_fix_delta, integer direction) {
  const auto ng{zone->ngg};
  constexpr integer n_reconstruction_max =
      7 + MAX_SPEC_NUMBER + 4; // rho,u,v,w,p,Y_{1...Ns},(k,omega,z,z_prime),E,gamma
  real pv_l[n_reconstruction_max], pv_r[n_reconstruction_max];
  const integer i_shared = tid - 1 + ng;
  reconstruction<mix_model, turb_method>(pv, pv_l, pv_r, i_shared, zone, param);

  // The entropy fix delta may not need shared memory, which may be replaced by shuffle instructions.
  integer n_reconstruct{param->n_var};
  if constexpr (mix_model == MixtureModel::FL) {
    n_reconstruct += param->n_spec;
  }
  compute_entropy_fix_delta<mix_model>(&pv[i_shared * n_reconstruct], &metric[i_shared * 9], param, entropy_fix_delta,
                                       tid);

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
  real kx = 0.5 * (metric[i_shared * 9 + direction * 3] + metric[(i_shared + 1) * 9 + direction * 3]);
  real ky = 0.5 * (metric[i_shared * 9 + direction * 3 + 1] + metric[(i_shared + 1) * 9 + direction * 3 + 1]);
  real kz = 0.5 * (metric[i_shared * 9 + direction * 3 + 2] + metric[(i_shared + 1) * 9 + direction * 3 + 2]);
  const real gradK = std::sqrt(kx * kx + ky * ky + kz * kz);
  real Uk = kx * u + ky * v + kz * w;

  real characteristic[3]{Uk - gradK * c, Uk, Uk + gradK * c};
  // entropy fix
  const real entropy_fix_delta_ave{entropy_fix_delta[tid] + entropy_fix_delta[tid + 1]};
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

  real c1 = (gamma - 1) * (ek * dq[0] - u * dq[1] - v * dq[2] - w * dq[3] + dq[4]) / c * c;
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
  LDq[2] = ky * c3 - ((kx * w - kz * u) * dq[0] - kz * dq[3] + kz * dq[1]) / c;
  LDq[3] = kz * c3 - ((ky * u - kx * v) * dq[0] - ky * dq[1] + kx * dq[2]) / c;
  LDq[4] = 0.5 * (c1 + c2);
  for (integer l = 0; l < param->n_spec; ++l) {
    LDq[l + 5] = dq[l + 5] - svm[l] * dq[0];
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

  fc[0] += 0.5 * c1;
  fc[1] += 0.5 * (u * c1 + kx * c2 - c * (kz * b[2] - ky * b[3]));
  fc[2] += 0.5 * (v * c1 + ky * c2 - c * (kx * b[3] - kz * b[1]));
  fc[3] += 0.5 * (w * c1 + kz * c2 - c * (ky * b[1] - kx * b[2]));
  fc[4] += 0.5 *
           (h * c1 + Uk * c2 - c * c * c0 / (gamma - 1) + c * (kz * v - ky * w) * b[1] + (kx * w - kz * u) * b[2] +
            (ky * u - kx * v) * b[3] + c3);
  for (integer l = 0; l < param->n_var - 5; ++l)
    fc[5 + l] += 0.5 * (b[l + 5] + svm[l] * c1);
}

template<MixtureModel mix_model>
__device__ void
compute_entropy_fix_delta(const real *pv, const real *metric, DParameter *param, real *entropy_fix_delta, integer tid) {
  const real U = abs(pv[1] * metric[0] + pv[2] * metric[1] + pv[3] * metric[2]);
  const real V = abs(pv[1] * metric[3] + pv[2] * metric[4] + pv[3] * metric[5]);
  const real W = abs(pv[1] * metric[6] + pv[2] * metric[7] + pv[3] * metric[8]);

  real specific_heat_ratio{gamma_air};
  if constexpr (mix_model != MixtureModel::Air) {
    real mw_inv{0.0};
    for (integer l = 0; l < param->n_spec; ++l) {
      mw_inv += pv[5 + l] / param->mw[l];
    }
    const real T{pv[4] / (pv[0] * R_u * mw_inv)};

    real cv{0}, cp{0};
    real cp_i[MAX_SPEC_NUMBER];
    compute_cp(T, cp_i, param);
    for (integer l = 0; l < param->n_spec; ++l) {
      cp += cp_i[l] * pv[5 + l];
      cv += pv[5 + l] * (cp_i[l] - R_u / param->mw[l]);
    }
    specific_heat_ratio = cp / cv;
  }

  const real c = std::sqrt(specific_heat_ratio * pv[4] / pv[0]);

  const real kx = sqrt(metric[0] * metric[0] + metric[1] * metric[1] + metric[2] * metric[2]);
  const real ky = sqrt(metric[3] * metric[3] + metric[4] * metric[4] + metric[5] * metric[5]);
  const real kz = sqrt(metric[6] * metric[6] + metric[7] * metric[7] + metric[8] * metric[8]);

  // need to be given in setup files
  real entropy_fix_factor{0.125};
  if (param->dim == 2) {
    entropy_fix_delta[tid] = entropy_fix_factor * (U + V + c * 0.5 * (kx + ky));
  } else {
    // 3D
    entropy_fix_delta[tid] = entropy_fix_factor * (U + V + W + c * (kx + ky + kz) / 3.0);
  }
}

template<MixtureModel mixtureModel>
__device__ void
compute_half_sum_left_right_flux(real *pv_l, real *pv_r, DParameter *param, const real *jac, real *metric,
                                 integer i_shared, integer direction, real *fc) {
  real JacKx = jac[i_shared] * metric[i_shared * 9 + direction * 3];
  real JacKy = jac[i_shared] * metric[i_shared * 9 + direction * 3 + 1];
  real JacKz = jac[i_shared] * metric[i_shared * 9 + direction * 3 + 2];
  real Uk = pv_l[1] * JacKx + pv_l[2] * JacKy + pv_l[3] * JacKz;

  integer n_reconstruct{param->n_var};
  if constexpr (mixtureModel == MixtureModel::FL) {
    n_reconstruct += param->n_spec;
  }
  fc[0] = 0.5 * Uk * pv_l[0];
  fc[1] = 0.5 * (fc[0] * pv_l[1] + pv_l[4] * JacKx);
  fc[2] = 0.5 * (fc[0] * pv_l[2] + pv_l[4] * JacKy);
  fc[3] = 0.5 * (fc[0] * pv_l[3] + pv_l[4] * JacKz);
  fc[4] = 0.5 * Uk * (pv_l[4] + pv_l[n_reconstruct]);
  for (integer l = 5; l < param->n_var; ++l) {
    if constexpr (mixtureModel != MixtureModel::FL) {
      fc[l] = 0.5 * fc[0] * pv_l[l];
    } else {
      fc[l] = 0.5 * fc[0] * pv_l[l + param->n_spec];
    }
  }

  JacKx = jac[i_shared + 1] * metric[(i_shared + 1) * 9 + direction * 3];
  JacKy = jac[i_shared + 1] * metric[(i_shared + 1) * 9 + direction * 3 + 1];
  JacKz = jac[i_shared + 1] * metric[(i_shared + 1) * 9 + direction * 3 + 2];
  Uk = pv_r[1] * JacKx + pv_r[2] * JacKy + pv_r[3] * JacKz;

  real coeff = Uk * pv_r[0];
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

}
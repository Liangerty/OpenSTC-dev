#pragma once

#include "Define.h"
#include "Field.h"
#include "DParameter.cuh"
#include "Thermo.cuh"
#include "Constants.h"

namespace cfd {
struct DParameter;

__device__ void first_order_reconstruct(const real *pv, real *pv_l, real *pv_r, integer idx_shared, integer n_var);

__device__ void
MUSCL_reconstruct(const real *pv, real *pv_l, real *pv_r, integer idx_shared, integer n_var, integer limiter);

__device__ void
NND2_reconstruct(const real *pv, real *pv_l, real *pv_r, integer idx_shared, integer n_var, integer limiter);

template<MixtureModel mix_model>
__device__ void
reconstruction(real *pv, real *pv_l, real *pv_r, const integer idx_shared, DParameter *param) {
  auto n_var = param->n_var;
  if constexpr (mix_model == MixtureModel::FL) {
    n_var += param->n_spec;
  }
  switch (param->reconstruction) {
    case 2:
      MUSCL_reconstruct(pv, pv_l, pv_r, idx_shared, n_var, param->limiter);
      break;
    case 3:
      NND2_reconstruct(pv, pv_l, pv_r, idx_shared, n_var, param->limiter);
      break;
    default:
      first_order_reconstruct(pv, pv_l, pv_r, idx_shared, n_var);
  }
  if constexpr (mix_model != MixtureModel::Air) {
    real el = 0.5 * (pv_l[1] * pv_l[1] + pv_l[2] * pv_l[2] + pv_l[3] * pv_l[3]);
    real er = 0.5 * (pv_r[1] * pv_r[1] + pv_r[2] * pv_r[2] + pv_r[3] * pv_r[3]);
    const auto n_spec = param->n_spec;
    real mw_inv_l{0.0}, mw_inv_r{0.0};
    for (int l = 0; l < n_spec; ++l) {
      mw_inv_l += pv_l[5 + l] / param->mw[l];
      mw_inv_r += pv_r[5 + l] / param->mw[l];
    }
    const real t_l = pv_l[4] / (pv_l[0] * R_u * mw_inv_l);
    const real t_r = pv_r[4] / (pv_r[0] * R_u * mw_inv_r);

    real hl[MAX_SPEC_NUMBER], hr[MAX_SPEC_NUMBER], cpl_i[MAX_SPEC_NUMBER], cpr_i[MAX_SPEC_NUMBER];
    compute_enthalpy_and_cp(t_l, hl, cpl_i, param);
    compute_enthalpy_and_cp(t_r, hr, cpr_i, param);
    real cpl{0}, cpr{0}, cvl{0}, cvr{0};
    for (auto l = 0; l < n_spec; ++l) {
      cpl += cpl_i[l] * pv_l[l + 5];
      cpr += cpr_i[l] * pv_r[l + 5];
      cvl += pv_l[l + 5] * (cpl_i[l] - R_u / param->mw[l]);
      cvr += pv_r[l + 5] * (cpr_i[l] - R_u / param->mw[l]);
      el += hl[l] * pv_l[l + 5];
      er += hr[l] * pv_r[l + 5];
    }
    pv_l[n_var] = pv_l[0] * el - pv_l[4]; //total energy
    pv_r[n_var] = pv_r[0] * er - pv_r[4];

    pv_l[n_var + 1] = cpl / cvl; //specific heat ratio
    pv_r[n_var + 1] = cpr / cvr;
  } else {
    real el = 0.5 * (pv_l[1] * pv_l[1] + pv_l[2] * pv_l[2] + pv_l[3] * pv_l[3]);
    real er = 0.5 * (pv_r[1] * pv_r[1] + pv_r[2] * pv_r[2] + pv_r[3] * pv_r[3]);
    pv_l[n_var] = el * pv_l[0] + pv_l[4] / (gamma_air - 1);
    pv_r[n_var] = er * pv_r[0] + pv_r[4] / (gamma_air - 1);
  }
}

} // cfd

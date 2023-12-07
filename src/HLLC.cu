#include "AWENO.cuh"
#include "HLLC.cuh"
#include "DParameter.cuh"
#include "Constants.h"
#include "Thermo.cuh"
#include "Parallel.h"

namespace cfd {

template<MixtureModel mix_model>
__device__ void
compute_hllc_flux(const real *pv_l, const real *pv_r, DParameter *param, integer tid, const real *metric,
                  const real *jac, real *fc, integer i_shared) {

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

  real gamma_l{gamma_air}, gamma_r{gamma_air};
  if constexpr (mix_model != MixtureModel::Air) {
    gamma_l = pv_l[n_reconstruct + 1];
    gamma_r = pv_r[n_reconstruct + 1];
  }
  const real Ul{kx * pv_l[1] + ky * pv_l[2] + kz * pv_l[3]};
  const real cl{std::sqrt(gamma_l * pv_l[4] / pv_l[0])};
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
  const real cr{std::sqrt(gamma_r * pv_r[4] / pv_r[0])};
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
    const real pCoeff{1.0 / (sl - sm)};
    const real QlCoeff{jac_ave * pCoeff * sm * (sl * gradK - Ul) * pv_l[0]};
    fci[0] = QlCoeff;
    const real dP{(sl * pm - sm * pv_l[4]) * pCoeff * jac_ave};
    fci[1] = QlCoeff * pv_l[1] + dP * kx;
    fci[2] = QlCoeff * pv_l[2] + dP * ky;
    fci[3] = QlCoeff * pv_l[3] + dP * kz;
    fci[4] = QlCoeff * pv_l[n_reconstruct] / pv_l[0] + pCoeff * jac_ave * (sl * pm * sm * gradK - sm * pv_l[4] * Ul);
    for (integer l = 5; l < n_var; ++l) {
      fci[l] = QlCoeff * pv_l[l];
    }
  } else {
    // Right star region, F_{*R}
    const real pCoeff{1.0 / (sr - sm)};
    const real QrCoeff{jac_ave * pCoeff * sm * (sr * gradK - Ur) * pv_r[0]};
    fci[0] = QrCoeff;
    const real dP{(sr * pm - sm * pv_r[4]) * pCoeff * jac_ave};
    fci[1] = QrCoeff * pv_r[1] + dP * kx;
    fci[2] = QrCoeff * pv_r[2] + dP * ky;
    fci[3] = QrCoeff * pv_r[3] + dP * kz;
    fci[4] = QrCoeff * pv_r[n_reconstruct] / pv_r[0] + pCoeff * jac_ave * (sr * pm * sm * gradK - sm * pv_r[4] * Ur);
    for (integer l = 5; l < n_var; ++l) {
      fci[l] = QrCoeff * pv_r[l];
    }
  }
}

template __device__ void
compute_hllc_flux<MixtureModel::Air>(const real *pv_l, const real *pv_r, DParameter *param, integer tid,
                                     const real *metric, const real *jac, real *fc, integer i_shared);
template __device__ void
compute_hllc_flux<MixtureModel::Mixture>(const real *pv_l, const real *pv_r, DParameter *param, integer tid,
                                     const real *metric, const real *jac, real *fc, integer i_shared);
template<> __device__ void
compute_hllc_flux<MixtureModel::MixtureFraction>(const real *pv_l, const real *pv_r, DParameter *param, integer tid,
                                     const real *metric, const real *jac, real *fc, integer i_shared) {
  printf("compute_hllc_flux<MixtureModel::MixtureFraction> is not implemented yet.\n");
}
template __device__ void
compute_hllc_flux<MixtureModel::FR>(const real *pv_l, const real *pv_r, DParameter *param, integer tid,
                                     const real *metric, const real *jac, real *fc, integer i_shared);
template<> __device__ void
compute_hllc_flux<MixtureModel::FL>(const real *pv_l, const real *pv_r, DParameter *param, integer tid,
                                     const real *metric, const real *jac, real *fc, integer i_shared) {
  printf("compute_hllc_flux<MixtureModel::FL> is not implemented yet.\n");
}
}

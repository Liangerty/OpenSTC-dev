#include "LaxFriedrichs.cuh"
#include "Constants.h"
#include <cmath>
#include "Parallel.h"

namespace cfd {
template<MixtureModel mix_model>
__device__ void
compute_lf_flux(const real *pv_l, const real *pv_r, DParameter *param, integer tid, const real *metric,
                const real *jac, real *fc, integer i_shared) {

}
template<>
__device__ void
compute_lf_flux<MixtureModel::Air>(const real *pv_l, const real *pv_r, DParameter *param, integer tid, const real *metric,
                const real *jac, real *fc, integer i_shared) {

  const integer n_var = param->n_var;
  integer n_reconstruct{n_var};

  // The metrics are just the average of the two adjacent cells.
  real kx = 0.5 * (metric[i_shared * 3] + metric[(i_shared + 1) * 3]);
  real ky = 0.5 * (metric[i_shared * 3 + 1] + metric[(i_shared + 1) * 3 + 1]);
  real kz = 0.5 * (metric[i_shared * 3 + 2] + metric[(i_shared + 1) * 3 + 2]);
  const real gradK = std::sqrt(kx * kx + ky * ky + kz * kz);

  // compute the left and right contravariance velocity
  const real Ukl{pv_l[1] * kx + pv_l[2] * ky + pv_l[3] * kz};
  const real Ukr{pv_r[1] * kx + pv_r[2] * ky + pv_r[3] * kz};
  const real cl{std::sqrt(gamma_air * pv_l[4] / pv_l[0])};
  const real cr{std::sqrt(gamma_air * pv_r[4] / pv_r[0])};
  const real spectral_radius{max(std::abs(Ukl) + cl * gradK, std::abs(Ukr) + cr * gradK)};

  auto fci = &fc[tid * n_var];
  const real half_jac_ave{0.5 * 0.5 * (jac[i_shared] + jac[i_shared + 1])};

  const real rhoUl{pv_l[0] * Ukl};
  const real rhoUr{pv_r[0] * Ukr};

  fci[0] = (rhoUl + rhoUr - spectral_radius * (pv_r[0] - pv_l[0])) * half_jac_ave;
  fci[1] = (rhoUl * pv_l[1] + rhoUr * pv_r[1] + kx * (pv_l[4] + pv_r[4]) -
            spectral_radius * (pv_r[1] * pv_r[0] - pv_l[1] * pv_l[0])) * half_jac_ave;
  fci[2] = (rhoUl * pv_l[2] + rhoUr * pv_r[2] + ky * (pv_l[4] + pv_r[4]) -
            spectral_radius * (pv_r[2] * pv_r[0] - pv_l[2] * pv_l[0])) * half_jac_ave;
  fci[3] = (rhoUl * pv_l[3] + rhoUr * pv_r[3] + kz * (pv_l[4] + pv_r[4]) -
            spectral_radius * (pv_r[3] * pv_r[0] - pv_l[3] * pv_l[0])) * half_jac_ave;
  fci[4] = ((pv_l[n_reconstruct] + pv_l[4]) * Ukl + (pv_r[n_reconstruct] + pv_r[4]) * Ukr -
            spectral_radius * (pv_r[n_reconstruct] - pv_l[n_reconstruct])) * half_jac_ave;
}

template __device__ void
compute_lf_flux<MixtureModel::Mixture>(const real *pv_l, const real *pv_r, DParameter *param, integer tid,
                                       const real *metric, const real *jac, real *fc, integer i_shared);

template __device__ void
compute_lf_flux<MixtureModel::FR>(const real *pv_l, const real *pv_r, DParameter *param, integer tid,
                                       const real *metric, const real *jac, real *fc, integer i_shared);

template __device__ void
compute_lf_flux<MixtureModel::FL>(const real *pv_l, const real *pv_r, DParameter *param, integer tid,
                                       const real *metric, const real *jac, real *fc, integer i_shared);

template __device__ void
compute_lf_flux<MixtureModel::MixtureFraction>(const real *pv_l, const real *pv_r, DParameter *param, integer tid,
                                       const real *metric, const real *jac, real *fc, integer i_shared);
}
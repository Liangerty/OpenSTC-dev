#include "AusmP.cuh"

namespace cfd{
template<MixtureModel mix_model>
__device__ void
compute_ausmPlus_flux(const real *pv_l, const real *pv_r, DParameter *param, integer tid, const real *metric,
                      const real *jac, real *fc, integer i_shared) {
  auto metric_l = &metric[i_shared * 3], metric_r = &metric[(i_shared + 1) * 3];
  auto jac_l = jac[i_shared], jac_r = jac[i_shared + 1];
  const real k1 = 0.5 * (jac_l * metric_l[0] + jac_r * metric_r[0]);
  const real k2 = 0.5 * (jac_l * metric_l[1] + jac_r * metric_r[1]);
  const real k3 = 0.5 * (jac_l * metric_l[2] + jac_r * metric_r[2]);
  const real grad_k_div_jac = std::sqrt(k1 * k1 + k2 * k2 + k3 * k3);

  const real ul = (k1 * pv_l[1] + k2 * pv_l[2] + k3 * pv_l[3]) / grad_k_div_jac;
  const real ur = (k1 * pv_r[1] + k2 * pv_r[2] + k3 * pv_r[3]) / grad_k_div_jac;

  const real pl = pv_l[4], pr = pv_r[4], rho_l = pv_l[0], rho_r = pv_r[0];
  const integer n_spec = param->n_spec;
  real gam_l{gamma_air}, gam_r{gamma_air};
  const integer n_var = param->n_var;
  auto n_reconstruct{n_var};
  if constexpr (mix_model == MixtureModel::FL) {
    n_reconstruct += n_spec;
  }
  if constexpr (mix_model != MixtureModel::Air) {
    gam_l = pv_l[n_reconstruct + 1];
    gam_r = pv_r[n_reconstruct + 1];
  }
  const real c = 0.5 * (std::sqrt(gam_l * pl / rho_l) + std::sqrt(gam_r * pr / rho_r));
  const real mach_l = ul / c, mach_r = ur / c;
  real mlp, mrn, plp, prn; // m for M, l/r for L/R, p/n for +/-. mlp=M_L^+
  constexpr static real alpha{3 / 16.0};

  if (std::abs(mach_l) > 1) {
    mlp = 0.5 * (mach_l + std::abs(mach_l));
    plp = mlp / mach_l;
  } else {
    const real ml_plus1_squared_div4 = (mach_l + 1) * (mach_l + 1) * 0.25;
    const real ml_squared_minus_1_squared = (mach_l * mach_l - 1) * (mach_l * mach_l - 1);
    mlp = ml_plus1_squared_div4 + 0.125 * ml_squared_minus_1_squared;
    plp = ml_plus1_squared_div4 * (2 - mach_l) + alpha * mach_l * ml_squared_minus_1_squared;
  }
  if (std::abs(mach_r) > 1) {
    mrn = 0.5 * (mach_r - std::abs(mach_r));
    prn = mrn / mach_r;
  } else {
    const real mr_minus1_squared_div4 = (mach_r - 1) * (mach_r - 1) * 0.25;
    const real mr_squared_minus_1_squared = (mach_r * mach_r - 1) * (mach_r * mach_r - 1);
    mrn = -mr_minus1_squared_div4 - 0.125 * mr_squared_minus_1_squared;
    prn = mr_minus1_squared_div4 * (2 + mach_r) - alpha * mach_r * mr_squared_minus_1_squared;
  }

  const real p_coeff = plp * pl + prn * pr;

  const real m_half = mlp + mrn;
  const real mach_pos = 0.5 * (m_half + std::abs(m_half));
  const real mach_neg = 0.5 * (m_half - std::abs(m_half));
  const real mass_flux_half = c * (rho_l * mach_pos + rho_r * mach_neg);
  const real coeff = mass_flux_half * grad_k_div_jac;

  auto fci = &fc[tid * n_var];
  if (mass_flux_half >= 0) {
    fci[0] = coeff;
    fci[1] = coeff * pv_l[1] + p_coeff * k1;
    fci[2] = coeff * pv_l[2] + p_coeff * k2;
    fci[3] = coeff * pv_l[3] + p_coeff * k3;
    fci[4] = coeff * (pv_l[n_reconstruct] + pv_l[4]) / pv_l[0];
    for (int l = 5; l < n_var; ++l) {
      if constexpr (mix_model != MixtureModel::FL) {
        fci[l] = coeff * pv_l[l];
      } else {
        // Flamelet model
        fci[l] = coeff * pv_l[l + n_spec];
      }
    }
  } else {
    fci[0] = coeff;
    fci[1] = coeff * pv_r[1] + p_coeff * k1;
    fci[2] = coeff * pv_r[2] + p_coeff * k2;
    fci[3] = coeff * pv_r[3] + p_coeff * k3;
    fci[4] = coeff * (pv_r[n_reconstruct] + pv_r[4]) / pv_r[0];
    for (int l = 5; l < n_var; ++l) {
      if constexpr (mix_model != MixtureModel::FL) {
        fci[l] = coeff * pv_r[l];
      } else {
        // Flamelet model
        fci[l] = coeff * pv_r[l + n_spec];
      }
    }
  }
}

template __device__ void
compute_ausmPlus_flux<MixtureModel::Air>(const real *pv_l, const real *pv_r, DParameter *param, integer tid, const real *metric,
                                         const real *jac, real *fc, integer i_shared);
template __device__ void
compute_ausmPlus_flux<MixtureModel::Mixture>(const real *pv_l, const real *pv_r, DParameter *param, integer tid, const real *metric,
                                         const real *jac, real *fc, integer i_shared);
template __device__ void
compute_ausmPlus_flux<MixtureModel::MixtureFraction>(const real *pv_l, const real *pv_r, DParameter *param, integer tid, const real *metric,
                                         const real *jac, real *fc, integer i_shared);
template __device__ void
compute_ausmPlus_flux<MixtureModel::FR>(const real *pv_l, const real *pv_r, DParameter *param, integer tid, const real *metric,
                                         const real *jac, real *fc, integer i_shared);
template __device__ void
compute_ausmPlus_flux<MixtureModel::FL>(const real *pv_l, const real *pv_r, DParameter *param, integer tid, const real *metric,
                                         const real *jac, real *fc, integer i_shared);

}
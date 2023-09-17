#pragma once

#include "Define.h"
#include "Thermo.cuh"
#include "Constants.h"
#include <cmath>
#include "SST.cuh"

namespace cfd {

struct DParameter;

template<MixtureModel mix_model, TurbMethod turb_method>
__device__ void compute_fv_2nd_order(const integer idx[3], DZone *zone, real *fv, DParameter *param);

template<MixtureModel mix_model, TurbMethod turb_method>
__device__ void compute_gv_2nd_order(const integer idx[3], DZone *zone, real *fv, DParameter *param);

template<MixtureModel mix_model, TurbMethod turb_method>
__device__ void compute_hv_2nd_order(const integer idx[3], DZone *zone, real *fv, DParameter *param);

template<MixtureModel mix_model, TurbMethod turb_method>
__device__ void compute_fv_2nd_order(const integer idx[3], DZone *zone, real *fv, DParameter *param) {
  const auto i = idx[0], j = idx[1], k = idx[2];
  const auto &m = zone->metric(i, j, k);
  const auto &m1 = zone->metric(i + 1, j, k);

  const real xi_x = 0.5 * (m(1, 1) + m1(1, 1));
  const real xi_y = 0.5 * (m(1, 2) + m1(1, 2));
  const real xi_z = 0.5 * (m(1, 3) + m1(1, 3));
  const real eta_x = 0.5 * (m(2, 1) + m1(2, 1));
  const real eta_y = 0.5 * (m(2, 2) + m1(2, 2));
  const real eta_z = 0.5 * (m(2, 3) + m1(2, 3));
  const real zeta_x = 0.5 * (m(3, 1) + m1(3, 1));
  const real zeta_y = 0.5 * (m(3, 2) + m1(3, 2));
  const real zeta_z = 0.5 * (m(3, 3) + m1(3, 3));

  // 1st order partial derivative of velocity to computational coordinate
  const auto &pv = zone->bv;
  const real u_xi = pv(i + 1, j, k, 1) - pv(i, j, k, 1);
  const real u_eta = 0.25 * (pv(i, j + 1, k, 1) - pv(i, j - 1, k, 1) + pv(i + 1, j + 1, k, 1) - pv(i + 1, j - 1, k, 1));
  const real u_zeta =
      0.25 * (pv(i, j, k + 1, 1) - pv(i, j, k - 1, 1) + pv(i + 1, j, k + 1, 1) - pv(i + 1, j, k - 1, 1));
  const real v_xi = pv(i + 1, j, k, 2) - pv(i, j, k, 2);
  const real v_eta = 0.25 * (pv(i, j + 1, k, 2) - pv(i, j - 1, k, 2) + pv(i + 1, j + 1, k, 2) - pv(i + 1, j - 1, k, 2));
  const real v_zeta =
      0.25 * (pv(i, j, k + 1, 2) - pv(i, j, k - 1, 2) + pv(i + 1, j, k + 1, 2) - pv(i + 1, j, k - 1, 2));
  const real w_xi = pv(i + 1, j, k, 3) - pv(i, j, k, 3);
  const real w_eta = 0.25 * (pv(i, j + 1, k, 3) - pv(i, j - 1, k, 3) + pv(i + 1, j + 1, k, 3) - pv(i + 1, j - 1, k, 3));
  const real w_zeta =
      0.25 * (pv(i, j, k + 1, 3) - pv(i, j, k - 1, 3) + pv(i + 1, j, k + 1, 3) - pv(i + 1, j, k - 1, 3));
  const real t_xi = pv(i + 1, j, k, 5) - pv(i, j, k, 5);
  const real t_eta = 0.25 * (pv(i, j + 1, k, 5) - pv(i, j - 1, k, 5) + pv(i + 1, j + 1, k, 5) - pv(i + 1, j - 1, k, 5));
  const real t_zeta =
      0.25 * (pv(i, j, k + 1, 5) - pv(i, j, k - 1, 5) + pv(i + 1, j, k + 1, 5) - pv(i + 1, j, k - 1, 5));

  // chain rule for derivative
  const real u_x = u_xi * xi_x + u_eta * eta_x + u_zeta * zeta_x;
  const real u_y = u_xi * xi_y + u_eta * eta_y + u_zeta * zeta_y;
  const real u_z = u_xi * xi_z + u_eta * eta_z + u_zeta * zeta_z;
  const real v_x = v_xi * xi_x + v_eta * eta_x + v_zeta * zeta_x;
  const real v_y = v_xi * xi_y + v_eta * eta_y + v_zeta * zeta_y;
  const real v_z = v_xi * xi_z + v_eta * eta_z + v_zeta * zeta_z;
  const real w_x = w_xi * xi_x + w_eta * eta_x + w_zeta * zeta_x;
  const real w_y = w_xi * xi_y + w_eta * eta_y + w_zeta * zeta_y;
  const real w_z = w_xi * xi_z + w_eta * eta_z + w_zeta * zeta_z;

  const real mul = 0.5 * (zone->mul(i, j, k) + zone->mul(i + 1, j, k));
  real mut{0};
  if constexpr (turb_method == TurbMethod::RANS) {
    mut = 0.5 * (zone->mut(i, j, k) + zone->mut(i + 1, j, k));
  }
  const real viscosity = mul + mut;

  // Compute the viscous stress
  real tau_xx = viscosity * (4 * u_x - 2 * v_y - 2 * w_z) / 3.0;
  real tau_yy = viscosity * (4 * v_y - 2 * u_x - 2 * w_z) / 3.0;
  real tau_zz = viscosity * (4 * w_z - 2 * u_x - 2 * v_y) / 3.0;
  const real tau_xy = viscosity * (u_y + v_x);
  const real tau_xz = viscosity * (u_z + w_x);
  const real tau_yz = viscosity * (v_z + w_y);
  if constexpr (turb_method == TurbMethod::RANS) {
    if (param->rans_model == 2) {
      // SST
      const real twoThirdrhoKm = -2.0 / 3 * 0.5 * (pv(i, j, k, 0) * zone->sv(i, j, k, param->n_spec) +
                                                   pv(i + 1, j, k, 0) * zone->sv(i + 1, j, k, param->n_spec));
      tau_xx += twoThirdrhoKm;
      tau_yy += twoThirdrhoKm;
      tau_zz += twoThirdrhoKm;
    }
  }

  const real xi_x_div_jac = 0.5 * (m(1, 1) * zone->jac(i, j, k) + m1(1, 1) * zone->jac(i + 1, j, k));
  const real xi_y_div_jac = 0.5 * (m(1, 2) * zone->jac(i, j, k) + m1(1, 2) * zone->jac(i + 1, j, k));
  const real xi_z_div_jac = 0.5 * (m(1, 3) * zone->jac(i, j, k) + m1(1, 3) * zone->jac(i + 1, j, k));

  fv[0] = 0;
  fv[1] = xi_x_div_jac * tau_xx + xi_y_div_jac * tau_xy + xi_z_div_jac * tau_xz;
  fv[2] = xi_x_div_jac * tau_xy + xi_y_div_jac * tau_yy + xi_z_div_jac * tau_yz;
  fv[3] = xi_x_div_jac * tau_xz + xi_y_div_jac * tau_yz + xi_z_div_jac * tau_zz;

  const real um = 0.5 * (pv(i, j, k, 1) + pv(i + 1, j, k, 1));
  const real vm = 0.5 * (pv(i, j, k, 2) + pv(i + 1, j, k, 2));
  const real wm = 0.5 * (pv(i, j, k, 3) + pv(i + 1, j, k, 3));
  const real t_x = t_xi * xi_x + t_eta * eta_x + t_zeta * zeta_x;
  const real t_y = t_xi * xi_y + t_eta * eta_y + t_zeta * zeta_y;
  const real t_z = t_xi * xi_z + t_eta * eta_z + t_zeta * zeta_z;
  real conductivity = 0.5 * (zone->thermal_conductivity(i, j, k) + zone->thermal_conductivity(i + 1, j, k));
  if constexpr (turb_method == TurbMethod::RANS) {
    conductivity += 0.5 * (zone->turb_therm_cond(i, j, k) + zone->turb_therm_cond(i + 1, j, k));
  }

  fv[4] = um * fv[1] + vm * fv[2] + wm * fv[3] +
          conductivity * (xi_x_div_jac * t_x + xi_y_div_jac * t_y + xi_z_div_jac * t_z);

  if constexpr (mix_model != MixtureModel::Air) {
    // Here, we only consider the influence of species diffusion.
    // That is, if we are solving mixture or finite rate, this part will compute the viscous term of species eqns and energy eqn.
    // If we are solving flamelet model, this part only contribute to the energy eqn.
    const integer n_spec{param->n_spec};
    const auto &y = zone->sv;

    real turb_diffusivity{0};
    if constexpr (turb_method == TurbMethod::RANS) {
      turb_diffusivity = mut / param->Sct;
    }

    real diffusivity[MAX_SPEC_NUMBER];
    real sum_GradXi_cdot_GradY_over_wl{0}, sum_rhoDkYk{0}, yk[MAX_SPEC_NUMBER];
    real CorrectionVelocityTerm{0};
    real mw_tot{0};
    real diffusion_driven_force[MAX_SPEC_NUMBER];
    for (int l = 0; l < n_spec; ++l) {
      yk[l] = 0.5 * (y(i, j, k, l) + y(i + 1, j, k, l));
      diffusivity[l] = 0.5 * (zone->rho_D(i, j, k, l) + zone->rho_D(i + 1, j, k, l)) + turb_diffusivity;

      const real y_xi = y(i + 1, j, k, l) - y(i, j, k, l);
      const real y_eta = 0.25 * (y(i, j + 1, k, l) - y(i, j - 1, k, l) + y(i + 1, j + 1, k, l) - y(i + 1, j - 1, k, l));
      const real y_zeta =
          0.25 * (y(i, j, k + 1, l) - y(i, j, k - 1, l) + y(i + 1, j, k + 1, l) - y(i + 1, j, k - 1, l));

      const real y_x = y_xi * xi_x + y_eta * eta_x + y_zeta * zeta_x;
      const real y_y = y_xi * xi_y + y_eta * eta_y + y_zeta * zeta_y;
      const real y_z = y_xi * xi_z + y_eta * eta_z + y_zeta * zeta_z;
      // Term 1, the gradient of mass fraction.
      const real GradXi_cdot_GradY = xi_x_div_jac * y_x + xi_y_div_jac * y_y + xi_z_div_jac * y_z;
      diffusion_driven_force[l] = GradXi_cdot_GradY;
      CorrectionVelocityTerm += diffusivity[l] * GradXi_cdot_GradY;

      // Term 2, the gradient of molecular weights, which is represented by sum of "gradient of mass fractions divided by molecular weight".
      sum_GradXi_cdot_GradY_over_wl += GradXi_cdot_GradY / param->mw[l];
      mw_tot += yk[l] / param->mw[l];
      sum_rhoDkYk += diffusivity[l] * yk[l];
    }
    mw_tot = 1.0 / mw_tot;
    CorrectionVelocityTerm -= mw_tot * sum_rhoDkYk * sum_GradXi_cdot_GradY_over_wl;

    // Term 3, diffusion caused by pressure gradient, and difference between Yk and Xk, which is more significant when the molecular weight is light.
    if (param->gradPInDiffusionFlux) {
      const real p_xi{pv(i + 1, j, k, 4) - pv(i, j, k, 4)};
      const real p_eta{
          0.25 * (pv(i, j + 1, k, 4) - pv(i, j - 1, k, 4) + pv(i + 1, j + 1, k, 4) - pv(i + 1, j - 1, k, 4))};
      const real p_zeta{
          0.25 * (pv(i, j, k + 1, 4) - pv(i, j, k - 1, 4) + pv(i + 1, j, k + 1, 4) - pv(i + 1, j, k - 1, 4))};

      const real p_x{p_xi * xi_x + p_eta * eta_x + p_zeta * zeta_x};
      const real p_y{p_xi * xi_y + p_eta * eta_y + p_zeta * zeta_y};
      const real p_z{p_xi * xi_z + p_eta * eta_z + p_zeta * zeta_z};

      const real gradXi_cdot_gradP_over_p{
          (xi_x_div_jac * p_x + xi_y_div_jac * p_y + xi_z_div_jac * p_z) /
          (0.5 * (pv(i + 1, j, k, 4) + pv(i, j, k, 4)))};

      // Velocity correction for the 3rd term
      for (int l = 0; l < n_spec; ++l) {
        diffusion_driven_force[l] += (mw_tot / param->mw[l] - 1) * yk[l] * gradXi_cdot_gradP_over_p;
        CorrectionVelocityTerm += (mw_tot / param->mw[l] - 1) * yk[l] * gradXi_cdot_gradP_over_p * diffusivity[l];
      }
    }

    real h[MAX_SPEC_NUMBER];
    const real tm = 0.5 * (pv(i, j, k, 5) + pv(i + 1, j, k, 5));
    compute_enthalpy(tm, h, param);

    for (int l = 0; l < n_spec; ++l) {
      const real diffusion_flux{
          diffusivity[l] * (diffusion_driven_force[l]
                            - mw_tot * yk[l] * sum_GradXi_cdot_GradY_over_wl)
          - yk[l] * CorrectionVelocityTerm};
      if constexpr (mix_model != MixtureModel::FL) {
        fv[5 + l] = diffusion_flux;
      }
      // Add the influence of species diffusion on total energy
      fv[4] += h[l] * diffusion_flux;
    }
  }

  if constexpr (turb_method == TurbMethod::RANS) {
    const integer rans_model{param->rans_model};

    switch (rans_model) {
      case 1: // SA
        break;
      case 2: // SST
      default:
        // Default SST
        const integer it = param->n_spec;
        auto &sv = zone->sv;

        const real k_xi = sv(i + 1, j, k, it) - sv(i, j, k, it);
        const real k_eta =
            0.25 * (sv(i, j + 1, k, it) - sv(i, j - 1, k, it) + sv(i + 1, j + 1, k, it) - sv(i + 1, j - 1, k, it));
        const real k_zeta =
            0.25 * (sv(i, j, k + 1, it) - sv(i, j, k - 1, it) + sv(i + 1, j, k + 1, it) - sv(i + 1, j, k - 1, it));

        const real k_x = k_xi * xi_x + k_eta * eta_x + k_zeta * zeta_x;
        const real k_y = k_xi * xi_y + k_eta * eta_y + k_zeta * zeta_y;
        const real k_z = k_xi * xi_z + k_eta * eta_z + k_zeta * zeta_z;

        const real omega_xi = sv(i + 1, j, k, it + 1) - sv(i, j, k, it + 1);
        const real omega_eta = 0.25 * (sv(i, j + 1, k, it + 1) - sv(i, j - 1, k, it + 1) + sv(i + 1, j + 1, k, it + 1) -
                                       sv(i + 1, j - 1, k, it + 1));
        const real omega_zeta = 0.25 *
                                (sv(i, j, k + 1, it + 1) - sv(i, j, k - 1, it + 1) + sv(i + 1, j, k + 1, it + 1) -
                                 sv(i + 1, j, k - 1, it + 1));

        const real omega_x = omega_xi * xi_x + omega_eta * eta_x + omega_zeta * zeta_x;
        const real omega_y = omega_xi * xi_y + omega_eta * eta_y + omega_zeta * zeta_y;
        const real omega_z = omega_xi * xi_z + omega_eta * eta_z + omega_zeta * zeta_z;

        const real wall_dist = 0.5 * (zone->wall_distance(i, j, k) + zone->wall_distance(i + 1, j, k));

        real f1{1};
        if (wall_dist > 1e-25) {
          const real km = 0.5 * (sv(i, j, k, it) + sv(i + 1, j, k, it));
          const real omegam = 0.5 * (sv(i, j, k, it + 1) + sv(i + 1, j, k, it + 1));
          const real param1{std::sqrt(km) / (0.09 * omegam * wall_dist)};

          const real rhom = 0.5 * (pv(i, j, k, 0) + pv(i + 1, j, k, 0));
          const real d2 = wall_dist * wall_dist;
          const real param2{500 * mul / (rhom * d2 * omegam)};
          const real CDkomega{max(1e-20, 2 * rhom * cfd::SST::sigma_omega2 / omegam *
                                         (k_x * omega_x + k_y * omega_y + k_z * omega_z))};
          const real param3{4 * rhom * SST::sigma_omega2 * km / (CDkomega * d2)};

          const real arg1{min(max(param1, param2), param3)};
          f1 = std::tanh(arg1 * arg1 * arg1 * arg1);
        }

        const real sigma_k = SST::sigma_k2 + SST::delta_sigma_k * f1;
        const real sigma_omega = SST::sigma_omega2 + SST::delta_sigma_omega * f1;

        const integer i_turb_cv{param->i_turb_cv};
        fv[i_turb_cv] = (mul + mut * sigma_k) * (xi_x_div_jac * k_x + xi_y_div_jac * k_y + xi_z_div_jac * k_z);
        fv[i_turb_cv + 1] =
            (mul + mut * sigma_omega) * (xi_x_div_jac * omega_x + xi_y_div_jac * omega_y + xi_z_div_jac * omega_z);
    }
  }

  if constexpr (mix_model == MixtureModel::FL) {
    // For flamelet model, we need to compute the viscous term for the mixture fraction and also its variance.
    const integer i_fl{param->i_fl}, i_fl_cv{param->i_fl_cv};
    const auto &sv = zone->sv;

    // First, compute the mixture fraction gradient
    const real mixFrac_xi = sv(i + 1, j, k, i_fl) - sv(i, j, k, i_fl);
    const real mixFrac_eta =
        0.25 * (sv(i, j + 1, k, i_fl) - sv(i, j - 1, k, i_fl) + sv(i + 1, j + 1, k, i_fl) - sv(i + 1, j - 1, k, i_fl));
    const real mixFrac_zeta =
        0.25 * (sv(i, j, k + 1, i_fl) - sv(i, j, k - 1, i_fl) + sv(i + 1, j, k + 1, i_fl) - sv(i + 1, j, k - 1, i_fl));

    const real mixFrac_x = mixFrac_xi * xi_x + mixFrac_eta * eta_x + mixFrac_zeta * zeta_x;
    const real mixFrac_y = mixFrac_xi * xi_y + mixFrac_eta * eta_y + mixFrac_zeta * zeta_y;
    const real mixFrac_z = mixFrac_xi * xi_z + mixFrac_eta * eta_z + mixFrac_zeta * zeta_z;

    const real rhoD{mul / param->Sc + mut / param->Sct};
    fv[i_fl_cv] = rhoD * (xi_x_div_jac * mixFrac_x + xi_y_div_jac * mixFrac_y + xi_z_div_jac * mixFrac_z);

    if constexpr (turb_method != TurbMethod::LES) {
      // For RANS / DES, we also solve the mixture fraction variance eqn.
      const real mixFracVar_xi = sv(i + 1, j, k, i_fl + 1) - pv(i, j, k, i_fl + 1);
      const real mixFracVar_eta = 0.25 * (sv(i, j + 1, k, i_fl + 1) - sv(i, j - 1, k, i_fl + 1) +
                                          sv(i + 1, j + 1, k, i_fl + 1) - sv(i + 1, j - 1, k, i_fl + 1));
      const real mixFracVar_zeta = 0.25 * (sv(i, j, k + 1, i_fl + 1) - sv(i, j, k - 1, i_fl + 1) +
                                           sv(i + 1, j, k + 1, i_fl + 1) - sv(i + 1, j, k - 1, i_fl + 1));

      const real mixFracVar_x = mixFracVar_xi * xi_x + mixFracVar_eta * eta_x + mixFracVar_zeta * zeta_x;
      const real mixFracVar_y = mixFracVar_xi * xi_y + mixFracVar_eta * eta_y + mixFracVar_zeta * zeta_y;
      const real mixFracVar_z = mixFracVar_xi * xi_z + mixFracVar_eta * eta_z + mixFracVar_zeta * zeta_z;

      fv[i_fl_cv + 1] =
          rhoD * (xi_x_div_jac * mixFracVar_x + xi_y_div_jac * mixFracVar_y + xi_z_div_jac * mixFracVar_z);
    }
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
__device__ void compute_gv_2nd_order(const integer *idx, DZone *zone, real *gv, cfd::DParameter *param) {
  const auto i = idx[0], j = idx[1], k = idx[2];
  const auto &m = zone->metric(i, j, k);
  const auto &m1 = zone->metric(i, j + 1, k);

  const real xi_x = 0.5 * (m(1, 1) + m1(1, 1));
  const real xi_y = 0.5 * (m(1, 2) + m1(1, 2));
  const real xi_z = 0.5 * (m(1, 3) + m1(1, 3));
  const real eta_x = 0.5 * (m(2, 1) + m1(2, 1));
  const real eta_y = 0.5 * (m(2, 2) + m1(2, 2));
  const real eta_z = 0.5 * (m(2, 3) + m1(2, 3));
  const real zeta_x = 0.5 * (m(3, 1) + m1(3, 1));
  const real zeta_y = 0.5 * (m(3, 2) + m1(3, 2));
  const real zeta_z = 0.5 * (m(3, 3) + m1(3, 3));

  // 1st order partial derivative of velocity to computational coordinate
  const auto &pv = zone->bv;
  const real u_xi = 0.25 * (pv(i + 1, j, k, 1) - pv(i - 1, j, k, 1) + pv(i + 1, j + 1, k, 1) - pv(i - 1, j + 1, k, 1));
  const real u_eta = pv(i, j + 1, k, 1) - pv(i, j, k, 1);
  const real u_zeta =
      0.25 * (pv(i, j, k + 1, 1) - pv(i, j, k - 1, 1) + pv(i, j + 1, k + 1, 1) - pv(i, j + 1, k - 1, 1));
  const real v_xi = 0.25 * (pv(i + 1, j, k, 2) - pv(i - 1, j, k, 2) + pv(i + 1, j + 1, k, 2) - pv(i - 1, j + 1, k, 2));
  const real v_eta = pv(i, j + 1, k, 2) - pv(i, j, k, 2);
  const real v_zeta =
      0.25 * (pv(i, j, k + 1, 2) - pv(i, j, k - 1, 2) + pv(i, j + 1, k + 1, 2) - pv(i, j + 1, k - 1, 2));
  const real w_xi = 0.25 * (pv(i + 1, j, k, 3) - pv(i - 1, j, k, 3) + pv(i + 1, j + 1, k, 3) - pv(i - 1, j + 1, k, 3));
  const real w_eta = pv(i, j + 1, k, 3) - pv(i, j, k, 3);
  const real w_zeta =
      0.25 * (pv(i, j, k + 1, 3) - pv(i, j, k - 1, 3) + pv(i, j + 1, k + 1, 3) - pv(i, j + 1, k - 1, 3));
  const real t_xi = 0.25 * (pv(i + 1, j, k, 5) - pv(i - 1, j, k, 5) + pv(i + 1, j + 1, k, 5) - pv(i - 1, j + 1, k, 5));
  const real t_eta = pv(i, j + 1, k, 5) - pv(i, j, k, 5);
  const real t_zeta =
      0.25 * (pv(i, j, k + 1, 5) - pv(i, j, k - 1, 5) + pv(i, j + 1, k + 1, 5) - pv(i, j + 1, k - 1, 5));

  // chain rule for derivative
  const real u_x = u_xi * xi_x + u_eta * eta_x + u_zeta * zeta_x;
  const real u_y = u_xi * xi_y + u_eta * eta_y + u_zeta * zeta_y;
  const real u_z = u_xi * xi_z + u_eta * eta_z + u_zeta * zeta_z;
  const real v_x = v_xi * xi_x + v_eta * eta_x + v_zeta * zeta_x;
  const real v_y = v_xi * xi_y + v_eta * eta_y + v_zeta * zeta_y;
  const real v_z = v_xi * xi_z + v_eta * eta_z + v_zeta * zeta_z;
  const real w_x = w_xi * xi_x + w_eta * eta_x + w_zeta * zeta_x;
  const real w_y = w_xi * xi_y + w_eta * eta_y + w_zeta * zeta_y;
  const real w_z = w_xi * xi_z + w_eta * eta_z + w_zeta * zeta_z;

  const real mul = 0.5 * (zone->mul(i, j, k) + zone->mul(i, j + 1, k));
  real mut{0};
  if constexpr (turb_method == TurbMethod::RANS) {
    mut = 0.5 * (zone->mut(i, j, k) + zone->mut(i, j + 1, k));
  }
  const real viscosity = mul + mut;

  // Compute the viscous stress
  real tau_xx = viscosity * (4 * u_x - 2 * v_y - 2 * w_z) / 3.0;
  real tau_yy = viscosity * (4 * v_y - 2 * u_x - 2 * w_z) / 3.0;
  real tau_zz = viscosity * (4 * w_z - 2 * u_x - 2 * v_y) / 3.0;
  const real tau_xy = viscosity * (u_y + v_x);
  const real tau_xz = viscosity * (u_z + w_x);
  const real tau_yz = viscosity * (v_z + w_y);

  if constexpr (turb_method == TurbMethod::RANS) {
    if (param->rans_model == 2) {
      // SST
      const real twoThirdrhoKm = -2.0 / 3 * 0.5 * (pv(i, j, k, 0) * zone->sv(i, j, k, param->n_spec) +
                                                   pv(i, j + 1, k, 0) * zone->sv(i, j + 1, k, param->n_spec));
      tau_xx += twoThirdrhoKm;
      tau_yy += twoThirdrhoKm;
      tau_zz += twoThirdrhoKm;
    }
  }

  const real eta_x_div_jac = 0.5 * (m(2, 1) * zone->jac(i, j, k) + m1(2, 1) * zone->jac(i, j + 1, k));
  const real eta_y_div_jac = 0.5 * (m(2, 2) * zone->jac(i, j, k) + m1(2, 2) * zone->jac(i, j + 1, k));
  const real eta_z_div_jac = 0.5 * (m(2, 3) * zone->jac(i, j, k) + m1(2, 3) * zone->jac(i, j + 1, k));

  gv[0] = 0;
  gv[1] = eta_x_div_jac * tau_xx + eta_y_div_jac * tau_xy + eta_z_div_jac * tau_xz;
  gv[2] = eta_x_div_jac * tau_xy + eta_y_div_jac * tau_yy + eta_z_div_jac * tau_yz;
  gv[3] = eta_x_div_jac * tau_xz + eta_y_div_jac * tau_yz + eta_z_div_jac * tau_zz;

  const real um = 0.5 * (pv(i, j, k, 1) + pv(i, j + 1, k, 1));
  const real vm = 0.5 * (pv(i, j, k, 2) + pv(i, j + 1, k, 2));
  const real wm = 0.5 * (pv(i, j, k, 3) + pv(i, j + 1, k, 3));
  const real t_x = t_xi * xi_x + t_eta * eta_x + t_zeta * zeta_x;
  const real t_y = t_xi * xi_y + t_eta * eta_y + t_zeta * zeta_y;
  const real t_z = t_xi * xi_z + t_eta * eta_z + t_zeta * zeta_z;
  real conductivity = 0.5 * (zone->thermal_conductivity(i, j, k) + zone->thermal_conductivity(i, j + 1, k));
  if constexpr (turb_method == TurbMethod::RANS) {
    conductivity += 0.5 * (zone->turb_therm_cond(i, j, k) + zone->turb_therm_cond(i, j + 1, k));
  }

  gv[4] = um * gv[1] + vm * gv[2] + wm * gv[3] +
          conductivity * (eta_x_div_jac * t_x + eta_y_div_jac * t_y + eta_z_div_jac * t_z);

  if constexpr (mix_model != MixtureModel::Air) {
    // Here, we only consider the influence of species diffusion.
    // That is, if we are solving mixture or finite rate, this part will compute the viscous term of species eqns and energy eqn.
    // If we are solving flamelet model, this part only contribute to the energy eqn.
    const integer n_spec{param->n_spec};
    const auto &y = zone->sv;

    real turb_diffusivity{0};
    if constexpr (turb_method == TurbMethod::RANS) {
      turb_diffusivity = mut / param->Sct;
    }

    real diffusivity[MAX_SPEC_NUMBER];
    real sum_GradEta_cdot_GradY_over_wl{0}, sum_rhoDkYk{0}, yk[MAX_SPEC_NUMBER];
    real CorrectionVelocityTerm{0};
    real mw_tot{0};
    real diffusion_driven_force[MAX_SPEC_NUMBER];
    for (int l = 0; l < n_spec; ++l) {
      yk[l] = 0.5 * (y(i, j, k, l) + y(i, j + 1, k, l));
      diffusivity[l] = 0.5 * (zone->rho_D(i, j, k, l) + zone->rho_D(i, j + 1, k, l)) + turb_diffusivity;

      const real y_xi = 0.25 * (y(i + 1, j, k, l) - y(i - 1, j, k, l) + y(i + 1, j + 1, k, l) - y(i - 1, j + 1, k, l));
      const real y_eta = y(i, j + 1, k, l) - y(i, j, k, l);
      const real y_zeta =
          0.25 * (y(i, j, k + 1, l) - y(i, j, k - 1, l) + y(i, j + 1, k + 1, l) - y(i, j + 1, k - 1, l));

      const real y_x = y_xi * xi_x + y_eta * eta_x + y_zeta * zeta_x;
      const real y_y = y_xi * xi_y + y_eta * eta_y + y_zeta * zeta_y;
      const real y_z = y_xi * xi_z + y_eta * eta_z + y_zeta * zeta_z;
      // Term 1, the gradient of mass fraction.
      const real GradEta_cdot_GradY = eta_x_div_jac * y_x + eta_y_div_jac * y_y + eta_z_div_jac * y_z;
      diffusion_driven_force[l] = GradEta_cdot_GradY;
      CorrectionVelocityTerm += diffusivity[l] * GradEta_cdot_GradY;

      // Term 2, the gradient of molecular weights, which is represented by sum of "gradient of mass fractions divided by molecular weight".
      sum_GradEta_cdot_GradY_over_wl += GradEta_cdot_GradY / param->mw[l];
      mw_tot += yk[l] / param->mw[l];
      sum_rhoDkYk += diffusivity[l] * yk[l];
    }
    mw_tot = 1.0 / mw_tot;
    CorrectionVelocityTerm -= mw_tot * sum_rhoDkYk * sum_GradEta_cdot_GradY_over_wl;

    // Term 3, diffusion caused by pressure gradient, and difference between Yk and Xk, which is more significant when the molecular weight is light.
    if (param->gradPInDiffusionFlux) {
      const real p_xi =
          0.25 * (pv(i + 1, j, k, 4) - pv(i - 1, j, k, 4) + pv(i + 1, j + 1, k, 4) - pv(i - 1, j + 1, k, 4));
      const real p_eta = pv(i, j + 1, k, 4) - pv(i, j, k, 4);
      const real p_zeta =
          0.25 * (pv(i, j, k + 1, 4) - pv(i, j, k - 1, 4) + pv(i, j + 1, k + 1, 4) - pv(i, j + 1, k - 1, 4));

      const real p_x{p_xi * xi_x + p_eta * eta_x + p_zeta * zeta_x};
      const real p_y{p_xi * xi_y + p_eta * eta_y + p_zeta * zeta_y};
      const real p_z{p_xi * xi_z + p_eta * eta_z + p_zeta * zeta_z};

      const real gradEta_cdot_gradP_over_p{
          (eta_x_div_jac * p_x + eta_y_div_jac * p_y + eta_z_div_jac * p_z) /
          (0.5 * (pv(i, j, k, 4) + pv(i, j + 1, k, 4)))};

      // Velocity correction for the 3rd term
      for (int l = 0; l < n_spec; ++l) {
        diffusion_driven_force[l] += (mw_tot / param->mw[l] - 1) * yk[l] * gradEta_cdot_gradP_over_p;
        CorrectionVelocityTerm += (mw_tot / param->mw[l] - 1) * yk[l] * gradEta_cdot_gradP_over_p * diffusivity[l];
      }
    }

    real h[MAX_SPEC_NUMBER];
    const real tm = 0.5 * (pv(i, j, k, 5) + pv(i, j + 1, k, 5));
    compute_enthalpy(tm, h, param);

    for (int l = 0; l < n_spec; ++l) {
      const real diffusion_flux{
          diffusivity[l] * (diffusion_driven_force[l]
                            - mw_tot * yk[l] * sum_GradEta_cdot_GradY_over_wl)
          - yk[l] * CorrectionVelocityTerm};
      if constexpr (mix_model != MixtureModel::FL) {
        gv[5 + l] = diffusion_flux;
      }
      // Add the influence of species diffusion on total energy
      gv[4] += h[l] * diffusion_flux;
    }
  }

  if constexpr (turb_method == TurbMethod::RANS) {
    const integer rans_model{param->rans_model};

    switch (rans_model) {
      case 1: // SA
        break;
      case 2: // SST
      default:
        // Default SST
        const integer it = param->n_spec;
        auto &sv = zone->sv;

        const real k_xi =
            0.25 * (sv(i + 1, j, k, it) - sv(i - 1, j, k, it) + sv(i + 1, j + 1, k, it) - sv(i - 1, j + 1, k, it));
        const real k_eta = sv(i, j + 1, k, it) - sv(i, j, k, it);
        const real k_zeta =
            0.25 * (sv(i, j, k + 1, it) - sv(i, j, k - 1, it) + sv(i, j + 1, k + 1, it) - sv(i, j + 1, k - 1, it));

        const real k_x = k_xi * xi_x + k_eta * eta_x + k_zeta * zeta_x;
        const real k_y = k_xi * xi_y + k_eta * eta_y + k_zeta * zeta_y;
        const real k_z = k_xi * xi_z + k_eta * eta_z + k_zeta * zeta_z;

        const real omega_xi = 0.25 * (sv(i + 1, j, k, it + 1) - sv(i - 1, j, k, it + 1) + sv(i + 1, j + 1, k, it + 1) -
                                      sv(i - 1, j + 1, k, it + 1));
        const real omega_eta = sv(i, j + 1, k, it + 1) - sv(i, j, k, it + 1);
        const real omega_zeta = 0.25 *
                                (sv(i, j, k + 1, it + 1) - sv(i, j, k - 1, it + 1) + sv(i, j + 1, k + 1, it + 1) -
                                 sv(i, j + 1, k - 1, it + 1));

        const real omega_x = omega_xi * xi_x + omega_eta * eta_x + omega_zeta * zeta_x;
        const real omega_y = omega_xi * xi_y + omega_eta * eta_y + omega_zeta * zeta_y;
        const real omega_z = omega_xi * xi_z + omega_eta * eta_z + omega_zeta * zeta_z;

        const real wall_dist = 0.5 * (zone->wall_distance(i, j, k) + zone->wall_distance(i, j + 1, k));

        real f1{1};
        if (wall_dist > 1e-25) {
          const real km = 0.5 * (sv(i, j, k, it) + sv(i, j + 1, k, it));
          const real omegam = 0.5 * (sv(i, j, k, it + 1) + sv(i, j + 1, k, it + 1));
          const real param1{std::sqrt(km) / (0.09 * omegam * wall_dist)};

          const real rhom = 0.5 * (pv(i, j, k, 0) + pv(i, j + 1, k, 0));
          const real d2 = wall_dist * wall_dist;
          const real param2{500 * mul / (rhom * d2 * omegam)};
          const real CDkomega{max(1e-20, 2 * rhom * SST::sigma_omega2 / omegam *
                                         (k_x * omega_x + k_y * omega_y + k_z * omega_z))};
          const real param3{4 * rhom * SST::sigma_omega2 * km / (CDkomega * d2)};

          const real arg1{min(max(param1, param2), param3)};
          f1 = std::tanh(arg1 * arg1 * arg1 * arg1);
        }

        const real sigma_k = SST::sigma_k2 + SST::delta_sigma_k * f1;
        const real sigma_omega = SST::sigma_omega2 + SST::delta_sigma_omega * f1;

        const integer i_turb_cv{param->i_turb_cv};
        gv[i_turb_cv] = (mul + mut * sigma_k) * (eta_x_div_jac * k_x + eta_y_div_jac * k_y + eta_z_div_jac * k_z);
        gv[i_turb_cv + 1] =
            (mul + mut * sigma_omega) * (eta_x_div_jac * omega_x + eta_y_div_jac * omega_y + eta_z_div_jac * omega_z);
    }
  }

  if constexpr (mix_model == MixtureModel::FL) {
    // For flamelet model, we need to compute the viscous term for the mixture fraction and also its variance.
    const integer i_fl{param->i_fl}, i_fl_cv{param->i_fl_cv};
    const auto &sv = zone->sv;

    // First, compute the mixture fraction gradient
    const real mixFrac_xi =
        0.25 * (sv(i + 1, j, k, i_fl) - sv(i - 1, j, k, i_fl) + sv(i + 1, j + 1, k, i_fl) - sv(i - 1, j + 1, k, i_fl));
    const real mixFrac_eta = sv(i, j + 1, k, i_fl) - sv(i, j, k, i_fl);
    const real mixFrac_zeta =
        0.25 * (sv(i, j, k + 1, i_fl) - sv(i, j, k - 1, i_fl) + sv(i, j + 1, k + 1, i_fl) - sv(i, j + 1, k - 1, i_fl));

    const real mixFrac_x = mixFrac_xi * xi_x + mixFrac_eta * eta_x + mixFrac_zeta * zeta_x;
    const real mixFrac_y = mixFrac_xi * xi_y + mixFrac_eta * eta_y + mixFrac_zeta * zeta_y;
    const real mixFrac_z = mixFrac_xi * xi_z + mixFrac_eta * eta_z + mixFrac_zeta * zeta_z;

    const real rhoD{mul / param->Sc + mut / param->Sct};
    gv[i_fl_cv] = rhoD * (eta_x_div_jac * mixFrac_x + eta_y_div_jac * mixFrac_y + eta_z_div_jac * mixFrac_z);

    if constexpr (turb_method != TurbMethod::LES) {
      // For LES, we do not need to compute the variance of mixture fraction.
      const real mixFracVar_xi = 0.25 * (sv(i + 1, j, k, i_fl + 1) - sv(i - 1, j, k, i_fl + 1) +
                                         sv(i + 1, j + 1, k, i_fl + 1) - sv(i - 1, j + 1, k, i_fl + 1));
      const real mixFracVar_eta = sv(i, j + 1, k, i_fl + 1) - sv(i, j, k, i_fl + 1);
      const real mixFracVar_zeta = 0.25 * (sv(i, j, k + 1, i_fl + 1) - sv(i, j, k - 1, i_fl + 1) +
                                           sv(i, j + 1, k + 1, i_fl + 1) - sv(i, j + 1, k - 1, i_fl + 1));

      const real mixFracVar_x = mixFracVar_xi * xi_x + mixFracVar_eta * eta_x + mixFracVar_zeta * zeta_x;
      const real mixFracVar_y = mixFracVar_xi * xi_y + mixFracVar_eta * eta_y + mixFracVar_zeta * zeta_y;
      const real mixFracVar_z = mixFracVar_xi * xi_z + mixFracVar_eta * eta_z + mixFracVar_zeta * zeta_z;

      gv[i_fl_cv + 1] =
          rhoD * (eta_x_div_jac * mixFracVar_x + eta_y_div_jac * mixFracVar_y + eta_z_div_jac * mixFracVar_z);
    }
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
__device__ void compute_hv_2nd_order(const integer *idx, DZone *zone, real *hv, cfd::DParameter *param) {
  const auto i = idx[0], j = idx[1], k = idx[2];
  const auto &m = zone->metric(i, j, k);
  const auto &m1 = zone->metric(i, j, k + 1);

  const real xi_x = 0.5 * (m(1, 1) + m1(1, 1));
  const real xi_y = 0.5 * (m(1, 2) + m1(1, 2));
  const real xi_z = 0.5 * (m(1, 3) + m1(1, 3));
  const real eta_x = 0.5 * (m(2, 1) + m1(2, 1));
  const real eta_y = 0.5 * (m(2, 2) + m1(2, 2));
  const real eta_z = 0.5 * (m(2, 3) + m1(2, 3));
  const real zeta_x = 0.5 * (m(3, 1) + m1(3, 1));
  const real zeta_y = 0.5 * (m(3, 2) + m1(3, 2));
  const real zeta_z = 0.5 * (m(3, 3) + m1(3, 3));

  // 1st order partial derivative of velocity to computational coordinate
  const auto &pv = zone->bv;
  const real u_xi = 0.25 * (pv(i + 1, j, k, 1) - pv(i - 1, j, k, 1) + pv(i + 1, j, k + 1, 1) - pv(i - 1, j, k + 1, 1));
  const real u_eta = 0.25 * (pv(i, j + 1, k, 1) - pv(i, j - 1, k, 1) + pv(i, j + 1, k + 1, 1) - pv(i, j - 1, k + 1, 1));
  const real u_zeta = pv(i, j, k + 1, 1) - pv(i, j, k, 1);
  const real v_xi = 0.25 * (pv(i + 1, j, k, 2) - pv(i - 1, j, k, 2) + pv(i + 1, j, k + 1, 2) - pv(i - 1, j, k + 1, 2));
  const real v_eta = 0.25 * (pv(i, j + 1, k, 2) - pv(i, j - 1, k, 2) + pv(i, j + 1, k + 1, 2) - pv(i, j - 1, k + 1, 2));
  const real v_zeta = pv(i, j, k + 1, 2) - pv(i, j, k, 2);
  const real w_xi = 0.25 * (pv(i + 1, j, k, 3) - pv(i - 1, j, k, 3) + pv(i + 1, j, k + 1, 3) - pv(i - 1, j, k + 1, 3));
  const real w_eta = 0.25 * (pv(i, j + 1, k, 3) - pv(i, j - 1, k, 3) + pv(i, j + 1, k + 1, 3) - pv(i, j - 1, k + 1, 3));
  const real w_zeta = pv(i, j, k + 1, 3) - pv(i, j, k, 3);
  const real t_xi = 0.25 * (pv(i + 1, j, k, 5) - pv(i - 1, j, k, 5) + pv(i + 1, j, k + 1, 5) - pv(i - 1, j, k + 1, 5));
  const real t_eta = 0.25 * (pv(i, j + 1, k, 5) - pv(i, j - 1, k, 5) + pv(i, j + 1, k + 1, 5) - pv(i, j - 1, k + 1, 5));
  const real t_zeta = pv(i, j, k + 1, 5) - pv(i, j, k, 5);

  // chain rule for derivative
  const real u_x = u_xi * xi_x + u_eta * eta_x + u_zeta * zeta_x;
  const real u_y = u_xi * xi_y + u_eta * eta_y + u_zeta * zeta_y;
  const real u_z = u_xi * xi_z + u_eta * eta_z + u_zeta * zeta_z;
  const real v_x = v_xi * xi_x + v_eta * eta_x + v_zeta * zeta_x;
  const real v_y = v_xi * xi_y + v_eta * eta_y + v_zeta * zeta_y;
  const real v_z = v_xi * xi_z + v_eta * eta_z + v_zeta * zeta_z;
  const real w_x = w_xi * xi_x + w_eta * eta_x + w_zeta * zeta_x;
  const real w_y = w_xi * xi_y + w_eta * eta_y + w_zeta * zeta_y;
  const real w_z = w_xi * xi_z + w_eta * eta_z + w_zeta * zeta_z;

  const real mul = 0.5 * (zone->mul(i, j, k) + zone->mul(i, j, k + 1));
  real mut{0};
  if constexpr (turb_method == TurbMethod::RANS) {
    mut = 0.5 * (zone->mut(i, j, k) + zone->mut(i, j, k + 1));
  }
  const real viscosity = mul + mut;

  // Compute the viscous stress
  real tau_xx = viscosity * (4 * u_x - 2 * v_y - 2 * w_z) / 3.0;
  real tau_yy = viscosity * (4 * v_y - 2 * u_x - 2 * w_z) / 3.0;
  real tau_zz = viscosity * (4 * w_z - 2 * u_x - 2 * v_y) / 3.0;
  const real tau_xy = viscosity * (u_y + v_x);
  const real tau_xz = viscosity * (u_z + w_x);
  const real tau_yz = viscosity * (v_z + w_y);

  if constexpr (turb_method == TurbMethod::RANS) {
    if (param->rans_model == 2) {
      // SST
      const real twoThirdrhoKm = -2.0 / 3 * 0.5 * (pv(i, j, k, 0) * zone->sv(i, j, k, param->n_spec) +
                                                   pv(i, j, k + 1, 0) * zone->sv(i, j, k + 1, param->n_spec));
      tau_xx += twoThirdrhoKm;
      tau_yy += twoThirdrhoKm;
      tau_zz += twoThirdrhoKm;
    }
  }

  const real zeta_x_div_jac = 0.5 * (m(3, 1) * zone->jac(i, j, k) + m1(3, 1) * zone->jac(i, j, k + 1));
  const real zeta_y_div_jac = 0.5 * (m(3, 2) * zone->jac(i, j, k) + m1(3, 2) * zone->jac(i, j, k + 1));
  const real zeta_z_div_jac = 0.5 * (m(3, 3) * zone->jac(i, j, k) + m1(3, 3) * zone->jac(i, j, k + 1));

  hv[0] = 0;
  hv[1] = zeta_x_div_jac * tau_xx + zeta_y_div_jac * tau_xy + zeta_z_div_jac * tau_xz;
  hv[2] = zeta_x_div_jac * tau_xy + zeta_y_div_jac * tau_yy + zeta_z_div_jac * tau_yz;
  hv[3] = zeta_x_div_jac * tau_xz + zeta_y_div_jac * tau_yz + zeta_z_div_jac * tau_zz;

  const real um = 0.5 * (pv(i, j, k, 1) + pv(i, j, k + 1, 1));
  const real vm = 0.5 * (pv(i, j, k, 2) + pv(i, j, k + 1, 2));
  const real wm = 0.5 * (pv(i, j, k, 3) + pv(i, j, k + 1, 3));
  const real t_x = t_xi * xi_x + t_eta * eta_x + t_zeta * zeta_x;
  const real t_y = t_xi * xi_y + t_eta * eta_y + t_zeta * zeta_y;
  const real t_z = t_xi * xi_z + t_eta * eta_z + t_zeta * zeta_z;
  real conductivity = 0.5 * (zone->thermal_conductivity(i, j, k) + zone->thermal_conductivity(i, j, k + 1));
  if constexpr (turb_method == TurbMethod::RANS) {
    conductivity += 0.5 * (zone->turb_therm_cond(i, j, k) + zone->turb_therm_cond(i, j, k + 1));
  }

  hv[4] = um * hv[1] + vm * hv[2] + wm * hv[3] +
          conductivity * (zeta_x_div_jac * t_x + zeta_y_div_jac * t_y + zeta_z_div_jac * t_z);

  if constexpr (mix_model != MixtureModel::Air) {
    // Here, we only consider the influence of species diffusion.
    // That is, if we are solving mixture or finite rate, this part will compute the viscous term of species eqns and energy eqn.
    // If we are solving flamelet model, this part only contribute to the energy eqn.
    const integer n_spec{param->n_spec};
    const auto &y = zone->sv;

    real turb_diffusivity{0};
    if constexpr (turb_method == TurbMethod::RANS) {
      turb_diffusivity = mut / param->Sct;
    }

    real diffusivity[MAX_SPEC_NUMBER];
    real sum_GradZeta_cdot_GradY_over_wl{0}, sum_rhoDkYk{0}, yk[MAX_SPEC_NUMBER];
    real CorrectionVelocityTerm{0};
    real mw_tot{0};
    real diffusion_driven_force[MAX_SPEC_NUMBER];
    for (int l = 0; l < n_spec; ++l) {
      yk[l] = 0.5 * (y(i, j, k, l) + y(i, j, k + 1, l));
      diffusivity[l] = 0.5 * (zone->rho_D(i, j, k, l) + zone->rho_D(i, j, k + 1, l)) + turb_diffusivity;

      const real y_xi = 0.25 * (y(i + 1, j, k, l) - y(i - 1, j, k, l) + y(i + 1, j, k + 1, l) - y(i - 1, j, k + 1, l));
      const real y_eta = 0.25 * (y(i, j + 1, k, l) - y(i, j - 1, k, l) + y(i, j + 1, k + 1, l) - y(i, j - 1, k + 1, l));
      const real y_zeta = y(i, j, k + 1, l) - y(i, j, k, l);

      const real y_x = y_xi * xi_x + y_eta * eta_x + y_zeta * zeta_x;
      const real y_y = y_xi * xi_y + y_eta * eta_y + y_zeta * zeta_y;
      const real y_z = y_xi * xi_z + y_eta * eta_z + y_zeta * zeta_z;
      // Term 1, the gradient of mass fraction.
      const real GradZeta_cdot_GradY = zeta_x_div_jac * y_x + zeta_y_div_jac * y_y + zeta_z_div_jac * y_z;
      diffusion_driven_force[l] = GradZeta_cdot_GradY;
      CorrectionVelocityTerm += diffusivity[l] * GradZeta_cdot_GradY;

      // Term 2, the gradient of molecular weights, which is represented by sum of "gradient of mass fractions divided by molecular weight".
      sum_GradZeta_cdot_GradY_over_wl += GradZeta_cdot_GradY / param->mw[l];
      mw_tot += yk[l] / param->mw[l];
      sum_rhoDkYk += diffusivity[l] * yk[l];

    }
    mw_tot = 1.0 / mw_tot;
    CorrectionVelocityTerm -= mw_tot * sum_rhoDkYk * sum_GradZeta_cdot_GradY_over_wl;

    // Term 3, diffusion caused by pressure gradient, and difference between Yk and Xk, which is more significant when the molecular weight is light.
    if (param->gradPInDiffusionFlux) {
      const real p_xi =
          0.25 * (pv(i + 1, j, k, 4) - pv(i - 1, j, k, 4) + pv(i + 1, j, k + 1, 4) - pv(i - 1, j, k + 1, 4));
      const real p_eta =
          0.25 * (pv(i, j + 1, k, 4) - pv(i, j - 1, k, 4) + pv(i, j + 1, k + 1, 4) - pv(i, j - 1, k + 1, 4));
      const real p_zeta = pv(i, j, k + 1, 4) - pv(i, j, k, 4);

      const real p_x{p_xi * xi_x + p_eta * eta_x + p_zeta * zeta_x};
      const real p_y{p_xi * xi_y + p_eta * eta_y + p_zeta * zeta_y};
      const real p_z{p_xi * xi_z + p_eta * eta_z + p_zeta * zeta_z};

      const real gradZeta_cdot_gradP_over_p{
          (zeta_x_div_jac * p_x + zeta_y_div_jac * p_y + zeta_z_div_jac * p_z) /
          (0.5 * (pv(i, j, k, 4) + pv(i, j, k + 1, 4)))};

      // Velocity correction for the 3rd term
      for (int l = 0; l < n_spec; ++l) {
        diffusion_driven_force[l] += (mw_tot / param->mw[l] - 1) * yk[l] * gradZeta_cdot_gradP_over_p;
        CorrectionVelocityTerm += (mw_tot / param->mw[l] - 1) * yk[l] * gradZeta_cdot_gradP_over_p * diffusivity[l];
      }
    }

    real h[MAX_SPEC_NUMBER];
    const real tm = 0.5 * (pv(i, j, k, 5) + pv(i, j, k + 1, 5));
    compute_enthalpy(tm, h, param);

    for (int l = 0; l < n_spec; ++l) {
      const real diffusion_flux{
          diffusivity[l] * (diffusion_driven_force[l]
                            - mw_tot * yk[l] * sum_GradZeta_cdot_GradY_over_wl)
          - yk[l] * CorrectionVelocityTerm};
      if constexpr (mix_model != MixtureModel::FL) {
        hv[5 + l] = diffusion_flux;
      }
      // Add the influence of species diffusion on total energy
      hv[4] += h[l] * diffusion_flux;
    }
  }

  if constexpr (turb_method == TurbMethod::RANS) {
    const integer rans_model{param->rans_model};

    switch (rans_model) {
      case 1: // SA
        break;
      case 2: // SST
      default:
        // Default SST
        const integer it = param->n_spec;
        auto &sv = zone->sv;
        const real k_xi =
            0.25 * (sv(i + 1, j, k, it) - sv(i - 1, j, k, it) + sv(i + 1, j, k + 1, it) - sv(i - 1, j, k + 1, it));
        const real k_eta =
            0.25 * (sv(i, j + 1, k, it) - sv(i, j - 1, k, it) + sv(i, j + 1, k + 1, it) - sv(i, j - 1, k + 1, it));
        const real k_zeta = sv(i, j, k + 1, it) - sv(i, j, k, it);

        const real k_x = k_xi * xi_x + k_eta * eta_x + k_zeta * zeta_x;
        const real k_y = k_xi * xi_y + k_eta * eta_y + k_zeta * zeta_y;
        const real k_z = k_xi * xi_z + k_eta * eta_z + k_zeta * zeta_z;

        const real omega_xi = 0.25 * (sv(i + 1, j, k, it + 1) - sv(i - 1, j, k, it + 1) + sv(i + 1, j, k + 1, it + 1) -
                                      sv(i - 1, j, k + 1, it + 1));
        const real omega_eta = 0.25 * (sv(i, j + 1, k, it + 1) - sv(i, j - 1, k, it + 1) + sv(i, j + 1, k + 1, it + 1) -
                                       sv(i, j - 1, k + 1, it + 1));
        const real omega_zeta = sv(i, j, k + 1, it + 1) - sv(i, j, k, it + 1);

        const real omega_x = omega_xi * xi_x + omega_eta * eta_x + omega_zeta * zeta_x;
        const real omega_y = omega_xi * xi_y + omega_eta * eta_y + omega_zeta * zeta_y;
        const real omega_z = omega_xi * xi_z + omega_eta * eta_z + omega_zeta * zeta_z;

        const real wall_dist = 0.5 * (zone->wall_distance(i, j, k) + zone->wall_distance(i, j, k + 1));

        real f1{1};
        if (wall_dist > 1e-25) {
          const real km = 0.5 * (sv(i, j, k, it) + sv(i, j, k + 1, it));
          const real omegam = 0.5 * (sv(i, j, k, it + 1) + sv(i, j, k + 1, it + 1));
          const real param1{std::sqrt(km) / (0.09 * omegam * wall_dist)};

          const real rhom = 0.5 * (pv(i, j, k, 0) + pv(i, j, k + 1, 0));
          const real d2 = wall_dist * wall_dist;
          const real param2{500 * mul / (rhom * d2 * omegam)};
          const real CDkomega{max(1e-20, 2 * rhom * SST::sigma_omega2 / omegam *
                                         (k_x * omega_x + k_y * omega_y + k_z * omega_z))};
          const real param3{4 * rhom * SST::sigma_omega2 * km / (CDkomega * d2)};

          const real arg1{min(max(param1, param2), param3)};
          f1 = std::tanh(arg1 * arg1 * arg1 * arg1);
        }

        const real sigma_k = SST::sigma_k2 + SST::delta_sigma_k * f1;
        const real sigma_omega = SST::sigma_omega2 + SST::delta_sigma_omega * f1;

        const integer i_turb_cv{param->i_turb_cv};
        hv[i_turb_cv] = (mul + mut * sigma_k) * (zeta_x_div_jac * k_x + zeta_y_div_jac * k_y + zeta_z_div_jac * k_z);
        hv[i_turb_cv + 1] =
            (mul + mut * sigma_omega) *
            (zeta_x_div_jac * omega_x + zeta_y_div_jac * omega_y + zeta_z_div_jac * omega_z);
    }
  }

  if constexpr (mix_model == MixtureModel::FL) {
    // For flamelet model, we need to compute the viscous term for the mixture fraction and also its variance.
    const integer i_fl{param->i_fl}, i_fl_cv{param->i_fl_cv};
    const auto &sv = zone->sv;

    // First, compute the mixture fraction gradient
    const real mixFrac_xi =
        0.25 * (sv(i + 1, j, k, i_fl) - sv(i - 1, j, k, i_fl) + sv(i + 1, j, k + 1, i_fl) - sv(i - 1, j, k + 1, i_fl));
    const real mixFrac_eta =
        0.25 * (sv(i, j + 1, k, i_fl) - sv(i, j - 1, k, i_fl) + sv(i, j + 1, k + 1, i_fl) - sv(i, j - 1, k + 1, i_fl));
    const real mixFrac_zeta = sv(i, j, k + 1, i_fl) - sv(i, j, k, i_fl);

    const real mixFrac_x = mixFrac_xi * xi_x + mixFrac_eta * eta_x + mixFrac_zeta * zeta_x;
    const real mixFrac_y = mixFrac_xi * xi_y + mixFrac_eta * eta_y + mixFrac_zeta * zeta_y;
    const real mixFrac_z = mixFrac_xi * xi_z + mixFrac_eta * eta_z + mixFrac_zeta * zeta_z;

    const real rhoD{mul / param->Sc + mut / param->Sct};
    hv[i_fl_cv] = rhoD * (zeta_x_div_jac * mixFrac_x + zeta_y_div_jac * mixFrac_y + zeta_z_div_jac * mixFrac_z);

    if constexpr (turb_method != TurbMethod::LES) {
      // For LES, we do not need to compute the variance of mixture fraction.
      const real mixFracVar_xi = 0.25 * (sv(i + 1, j, k, i_fl + 1) - sv(i - 1, j, k, i_fl + 1) +
                                         sv(i + 1, j, k + 1, i_fl + 1) - sv(i - 1, j, k + 1, i_fl + 1));
      const real mixFracVar_eta = 0.25 * (sv(i, j + 1, k, i_fl + 1) - sv(i, j - 1, k, i_fl + 1) +
                                          sv(i, j + 1, k + 1, i_fl + 1) - sv(i, j - 1, k + 1, i_fl + 1));
      const real mixFracVar_zeta = sv(i, j, k + 1, i_fl + 1) - sv(i, j, k - 1, i_fl + 1);

      const real mixFracVar_x = mixFracVar_xi * xi_x + mixFracVar_eta * eta_x + mixFracVar_zeta * zeta_x;
      const real mixFracVar_y = mixFracVar_xi * xi_y + mixFracVar_eta * eta_y + mixFracVar_zeta * zeta_y;
      const real mixFracVar_z = mixFracVar_xi * xi_z + mixFracVar_eta * eta_z + mixFracVar_zeta * zeta_z;

      hv[i_fl_cv + 1] =
          rhoD * (zeta_x_div_jac * mixFracVar_x + zeta_y_div_jac * mixFracVar_y + zeta_z_div_jac * mixFracVar_z);
    }
  }
}

} // cfd

// Contains some operations on different variables.
// E.g., total energy is computed from V and h; conservative variables are computed from basic variables
#pragma once

#include "Define.h"
#include "Field.h"
#include "DParameter.cuh"
#include "Thermo.cuh"
#include "Constants.h"
#include "Transport.cuh"
#include "SST.cuh"
#include "FlameletLib.cuh"

namespace cfd {
__device__ void
compute_temperature_and_pressure(int i, int j, int k, const DParameter *param, DZone *zone, real total_energy);

//template<MixtureModel mixture_model>
//__device__ void compute_total_energy(integer i, integer j, integer k, cfd::DZone *zone, DParameter *param) {
//  auto &bv = zone->bv;
//  auto &vel = zone->vel;
//  auto &cv = zone->cv;
//  auto &sv = zone->sv;
//
//  vel(i, j, k) = bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3);
//  cv(i, j, k, 4) = 0.5 * vel(i, j, k);
//  if constexpr (mixture_model != MixtureModel::Air) {
//    real enthalpy[MAX_SPEC_NUMBER];
//    compute_enthalpy(bv(i, j, k, 5), enthalpy, param);
//    // Add species enthalpy together up to kinetic energy to get total enthalpy
//    for (auto l = 0; l < param->n_spec; l++) {
//      // h = \Sum_{i=1}^{n_spec} h_i * Y_i
//      cv(i, j, k, 4) += enthalpy[l] * sv(i, j, k, l);
//    }
//    cv(i, j, k, 4) *= bv(i, j, k, 0); // \rho * h
//    cv(i, j, k, 4) -= bv(i, j, k, 4); // (\rho e =\rho h - p)
//  } else {
//    cv(i, j, k, 4) += bv(i, j, k, 4) / (gamma_air - 1);
//  }
//  vel(i, j, k) = sqrt(vel(i, j, k));
//}

//template<MixtureModel mix_model, TurbMethod turb_method>
//__global__ void compute_cv_from_bv(DZone *zone, DParameter *param) {
//  const integer ngg{zone->ngg}, mx{zone->mx}, my{zone->my}, mz{zone->mz};
//  integer i = (integer) (blockDim.x * blockIdx.x + threadIdx.x) - ngg;
//  integer j = (integer) (blockDim.y * blockIdx.y + threadIdx.y) - ngg;
//  integer k = (integer) (blockDim.z * blockIdx.z + threadIdx.z) - ngg;
//  if (i >= mx + ngg || j >= my + ngg || k >= mz + ngg) return;
//
//  const auto &bv = zone->bv;
//  auto &cv = zone->cv;
//  const real rho = bv(i, j, k, 0);
//  const real u = bv(i, j, k, 1);
//  const real v = bv(i, j, k, 2);
//  const real w = bv(i, j, k, 3);
//
//  cv(i, j, k, 0) = rho;
//  cv(i, j, k, 1) = rho * u;
//  cv(i, j, k, 2) = rho * v;
//  cv(i, j, k, 3) = rho * w;
//  // It seems we don't need an if here, if there is no other scalars, n_scalar=0; else, n_scalar=n_spec+n_turb
//  const auto &sv = zone->sv;
//  if constexpr (mix_model != MixtureModel::FL) {
//    const integer n_scalar{param->n_scalar};
//    for (auto l = 0; l < n_scalar; ++l) {
//      cv(i, j, k, 5 + l) = rho * sv(i, j, k, l);
//    }
//  } else {
//    // Flamelet model
//    const integer n_spec{param->n_spec};
//    const integer n_scalar_transported{param->n_scalar_transported};
//    for (auto l = 0; l < n_scalar_transported; ++l) {
//      cv(i, j, k, 5 + l) = rho * sv(i, j, k, l + n_spec);
//    }
//  }
//
//  compute_total_energy<mix_model>(i, j, k, zone, param);
//}
template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void compute_velocity(DZone *zone, DParameter *param) {
  const integer ngg{zone->ngg}, mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = (integer) (blockDim.x * blockIdx.x + threadIdx.x) - ngg;
  integer j = (integer) (blockDim.y * blockIdx.y + threadIdx.y) - ngg;
  integer k = (integer) (blockDim.z * blockIdx.z + threadIdx.z) - ngg;
  if (i >= mx + ngg || j >= my + ngg || k >= mz + ngg) return;

  const auto &bv = zone->bv;

  zone->vel(i, j, k) = std::sqrt(bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3));
}

//template<MixtureModel mix_model, TurbMethod turb_method>
//__device__ void compute_cv_from_bv_1_point(DZone *zone, DParameter *param, integer i, integer j, integer k) {
//  const auto &bv = zone->bv;
//  auto &cv = zone->cv;
//  const real rho = bv(i, j, k, 0);
//
//  cv(i, j, k, 0) = rho;
//  cv(i, j, k, 1) = rho * bv(i, j, k, 1);
//  cv(i, j, k, 2) = rho * bv(i, j, k, 2);
//  cv(i, j, k, 3) = rho * bv(i, j, k, 3);
//  // It seems we don't need an if here, if there is no other scalars, n_scalar=0; else, n_scalar=n_spec+n_turb
//  const auto &sv = zone->sv;
//  if constexpr (mix_model != MixtureModel::FL) {
//    const integer n_scalar{param->n_scalar};
//    for (auto l = 0; l < n_scalar; ++l) {
//      cv(i, j, k, 5 + l) = rho * sv(i, j, k, l);
//    }
//  } else {
//    // Flamelet model
//    const integer n_spec{param->n_spec};
//    const integer n_scalar_transported{param->n_scalar_transported};
//    for (auto l = 0; l < n_scalar_transported; ++l) {
//      cv(i, j, k, 5 + l) = rho * sv(i, j, k, l + n_spec);
//    }
//  }
//
//  compute_total_energy<mix_model>(i, j, k, zone, param);
//}

template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void update_physical_properties(DZone *zone, DParameter *param) {
  const integer mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = (integer) (blockDim.x * blockIdx.x + threadIdx.x) - 1;
  integer j = (integer) (blockDim.y * blockIdx.y + threadIdx.y) - 1;
  integer k = (integer) (blockDim.z * blockIdx.z + threadIdx.z) - 1;
  if (i >= mx + 1 || j >= my + 1 || k >= mz + 1) return;

  const real temperature{zone->bv(i, j, k, 5)};
  if constexpr (mix_model != MixtureModel::Air) {
    const integer n_spec{param->n_spec};
    auto &yk = zone->sv;
    real mw{0}, cp_tot{0}, cv{0};
    real cp[MAX_SPEC_NUMBER];
    compute_cp(temperature, cp, param);
    for (auto l = 0; l < n_spec; ++l) {
      mw += yk(i, j, k, l) / param->mw[l];
      cp_tot += yk(i, j, k, l) * cp[l];
      cv += yk(i, j, k, l) * (cp[l] - R_u / param->mw[l]);
    }
    mw = 1 / mw;
    zone->cp(i, j, k) = cp_tot;
    zone->gamma(i, j, k) = cp_tot / cv;
    zone->acoustic_speed(i, j, k) = std::sqrt(zone->gamma(i, j, k) * R_u * temperature / mw);
    compute_transport_property(i, j, k, temperature, mw, cp, param, zone);
  } else {
    constexpr real c_temp{gamma_air * R_u / mw_air};
    constexpr real cp{gamma_air * R_u / mw_air / (gamma_air - 1)};
    const real pr = param->Pr;
    zone->acoustic_speed(i, j, k) = std::sqrt(c_temp * temperature);
    zone->mul(i, j, k) = Sutherland(temperature);
    zone->thermal_conductivity(i, j, k) = zone->mul(i, j, k) * cp / pr;
  }
  zone->mach(i, j, k) = zone->vel(i, j, k) / zone->acoustic_speed(i, j, k);
}

template<MixtureModel mix_model>
__global__ void initialize_mut(DZone *zone, DParameter *param) {
  const integer mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = (integer) (blockDim.x * blockIdx.x + threadIdx.x) - 1;
  integer j = (integer) (blockDim.y * blockIdx.y + threadIdx.y) - 1;
  integer k = (integer) (blockDim.z * blockIdx.z + threadIdx.z) - 1;
  if (i >= mx + 1 || j >= my + 1 || k >= mz + 1) return;

  switch (param->rans_model) {
    case 1://SA
      break;
    case 2:
    default: // SST
      const real temperature{zone->bv(i, j, k, 5)};
      real mul = Sutherland(temperature);
      if constexpr (mix_model != MixtureModel::Air) {
        auto &yk = zone->sv;
        real mw{0};
        for (auto l = 0; l < param->n_spec; ++l) {
          mw += yk(i, j, k, l) / param->mw[l];
        }
        mw = 1 / mw;
        mul = compute_viscosity(i, j, k, temperature, mw, param, zone);
      }
      SST::compute_mut(zone, i, j, k, mul, param);
  }
}

template<MixtureModel mixture_model>
__device__ real compute_total_energy_1_point(integer i, integer j, integer k, cfd::DZone *zone, DParameter *param) {
  auto &bv = zone->bv;
  auto &sv = zone->sv;

  const real V2 = bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3);
  real total_energy = 0.5 * V2;
  if constexpr (mixture_model != MixtureModel::Air) {
    real enthalpy[MAX_SPEC_NUMBER];
    compute_enthalpy(bv(i, j, k, 5), enthalpy, param);
    // Add species enthalpy together up to kinetic energy to get total enthalpy
    for (auto l = 0; l < param->n_spec; l++) {
      // h = \Sum_{i=1}^{n_spec} h_i * Y_i
      total_energy += enthalpy[l] * sv(i, j, k, l);
    }
    total_energy *= bv(i, j, k, 0); // \rho * h
    total_energy -= bv(i, j, k, 4); // (\rho e =\rho h - p)
  } else {
    total_energy += bv(i, j, k, 4) / (gamma_air - 1);
  }
  return total_energy;
}

template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void update_q_and_bv(cfd::DZone *zone, DParameter *param) {
  const integer extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  auto &bv = zone->bv;
  const auto &dq = zone->dq;

  // Record and compute some of the old quantities first
  const real density_n = bv(i, j, k, 0);
  const real total_energy_n = compute_total_energy_1_point<mix_model>(i, j, k, zone, param);

  real dt_div_jac = zone->dt_local(i, j, k) / zone->jac(i, j, k);
  // Update density first
  bv(i, j, k, 0) += dq(i, j, k, 0) * dt_div_jac;
  // Get the 1/density
  const real density_inv = 1.0 / bv(i, j, k, 0);
  // Update u, v, w
  bv(i, j, k, 1) = density_inv * (density_n * bv(i, j, k, 1) + dq(i, j, k, 1) * dt_div_jac);
  bv(i, j, k, 2) = density_inv * (density_n * bv(i, j, k, 2) + dq(i, j, k, 2) * dt_div_jac);
  if (extent[2] == 1) {
    bv(i, j, k, 3) = 0;
  } else {
    bv(i, j, k, 3) = density_inv * (density_n * bv(i, j, k, 3) + dq(i, j, k, 3) * dt_div_jac);
  }
  // Update total energy
  real total_energy = (total_energy_n + dq(i, j, k, 4) * dt_div_jac);

  // Update scalars
  auto &sv = zone->sv;
  if constexpr (mix_model != MixtureModel::FL) {
    // For multiple species or RANS methods, there will be scalars to be computed
    for (integer l = 0; l < param->n_scalar; ++l) {
      sv(i, j, k, l) = density_inv * (density_n * sv(i, j, k, l) + dq(i, j, k, 5 + l) * dt_div_jac);
    }
  } else {
    // Flamelet model
    const integer n_spec{param->n_spec};
    for (integer l = 0; l < param->n_scalar_transported; ++l) {
      sv(i, j, k, l + n_spec) = density_inv * (density_n * sv(i, j, k, l + n_spec) + dq(i, j, k, 5 + l) * dt_div_jac);
    }
    real yk_ave[MAX_SPEC_NUMBER];
    memset(yk_ave, 0, sizeof(real) * n_spec);
    compute_massFraction_from_MixtureFraction(zone, i, j, k, param, yk_ave);
    if (param->n_fl_step > 10000) {
      for (integer l = 0; l < n_spec; ++l) {
        sv(i, j, k, l) = yk_ave[l];
      }
    } else {
      for (integer l = 0; l < n_spec; ++l) {
        real yk_mix = param->yk_lib(l, 0, 0, 0) +
                      sv(i, j, k, param->i_fl) * (param->yk_lib(l, 0, 0, param->n_z) - param->yk_lib(l, 0, 0, 0));
        sv(i, j, k, l) = yk_mix + param->n_fl_step * (yk_ave[l] - yk_mix) / 10000.0;
      }
    }
  }

  // update temperature and pressure from total energy and species composition
  const real V2 =
      bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3); //V^2
  if constexpr (mix_model != MixtureModel::Air) {
    compute_temperature_and_pressure(i, j, k, param, zone, total_energy);
  } else {
    // Air
    bv(i, j, k, 4) = (gamma_air - 1) * (total_energy - 0.5 * bv(i, j, k, 0) * V2);
    bv(i, j, k, 5) = bv(i, j, k, 4) * mw_air * density_inv / R_u;
  }

  zone->vel(i, j, k) = std::sqrt(V2);
}

__global__ void eliminate_k_gradient(cfd::DZone *zone, const DParameter *param);

}
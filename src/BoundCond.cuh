#pragma once

#include "BoundCond.h"
#include "Mesh.h"
#include "Field.h"
#include "DParameter.cuh"
#include "FieldOperation.cuh"
#include "SST.cuh"
#include "Parallel.h"

namespace cfd {

struct BCInfo {
  integer label = 0;
  integer n_boundary = 0;
  int2 *boundary = nullptr;
};

struct DBoundCond {
  DBoundCond() = default;

  void initialize_bc_on_GPU(Mesh &mesh, std::vector<Field> &field, Species &species, Parameter &parameter);

  void link_bc_to_boundaries(Mesh &mesh, std::vector<Field> &field) const;

  template<MixtureModel mix_model, TurbulenceMethod turb_method>
  void apply_boundary_conditions(const Block &block, Field &field, DParameter *param) const;

  integer n_wall = 0, n_symmetry = 0, n_inflow = 0, n_outflow = 0, n_farfield = 0, n_subsonic_inflow = 0, n_back_pressure = 0;
  BCInfo *wall_info = nullptr;
  BCInfo *symmetry_info = nullptr;
  BCInfo *inflow_info = nullptr;
  BCInfo *outflow_info = nullptr;
  BCInfo *farfield_info = nullptr;
  BCInfo *subsonic_inflow_info = nullptr;
  BCInfo *back_pressure_info = nullptr;
  Wall *wall = nullptr;
  Symmetry *symmetry = nullptr;
  Inflow *inflow = nullptr;
  Outflow *outflow = nullptr;
  FarField *farfield = nullptr;
  SubsonicInflow *subsonic_inflow = nullptr;
  BackPressure *back_pressure = nullptr;
};

void count_boundary_of_type_bc(const std::vector<Boundary> &boundary, integer n_bc, integer **sep, integer blk_idx,
                               integer n_block, BCInfo *bc_info);

void link_boundary_and_condition(const std::vector<Boundary> &boundary, BCInfo *bc, integer n_bc, integer **sep,
                                 integer i_zone);

template<MixtureModel mix_model, TurbulenceMethod turb_method>
__global__ void apply_symmetry(DZone *zone, integer i_face, DParameter *param) {
  const auto &b = zone->boundary[i_face];
  auto range_start = b.range_start, range_end = b.range_end;
  integer i = range_start[0] + (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  integer j = range_start[1] + (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  integer k = range_start[2] + (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto face = b.face;
  integer dir[]{0, 0, 0};
  dir[face] = b.direction;

  const integer inner_idx[3]{i - dir[0], j - dir[1], k - dir[2]};

  auto metric = zone->metric(i, j, k);
  real k_x{metric(face + 1, 1)}, k_y{metric(face + 1, 2)}, k_z{metric(face + 1, 3)};
  real k_magnitude = sqrt(k_x * k_x + k_y * k_y + k_z * k_z);
  k_x /= k_magnitude;
  k_y /= k_magnitude;
  k_z /= k_magnitude;

  auto &bv = zone->bv;
  real u1{bv(inner_idx[0], inner_idx[1], inner_idx[2], 1)}, v1{bv(inner_idx[0], inner_idx[1], inner_idx[2], 2)}, w1{
      bv(inner_idx[0], inner_idx[1], inner_idx[2], 3)};
  real u_k{k_x * u1 + k_y * v1 + k_z * w1};
  const real u_t{u1 - k_x * u_k}, v_t{v1 - k_y * u_k}, w_t{w1 - k_z * u_k};

  // The gradient of tangential velocity should be zero.
  bv(i, j, k, 1) = u_t;
  bv(i, j, k, 2) = v_t;
  bv(i, j, k, 3) = w_t;
  zone->vel(i, j, k) = std::sqrt(u_t * u_t + v_t * v_t + w_t * w_t);
  // The gradient of pressure, density, and scalars should also be zero.
  bv(i, j, k, 0) = bv(inner_idx[0], inner_idx[1], inner_idx[2], 0);
  bv(i, j, k, 4) = bv(inner_idx[0], inner_idx[1], inner_idx[2], 4);
  bv(i, j, k, 5) = bv(inner_idx[0], inner_idx[1], inner_idx[2], 5);
  auto &sv = zone->sv;
  for (integer l = 0; l < param->n_scalar; ++l) {
    sv(i, j, k, l) = sv(inner_idx[0], inner_idx[1], inner_idx[2], l);
  }

  // For ghost grids
  for (integer g = 1; g <= zone->ngg; ++g) {
    const integer gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
    const integer ii{i - g * dir[0]}, ij{j - g * dir[1]}, ik{k - g * dir[2]};

    bv(gi, gj, gk, 0) = bv(ii, ij, ik, 0);

    auto &u{bv(ii, ij, ik, 1)}, v{bv(ii, ij, ik, 2)}, w{bv(ii, ij, ik, 3)};
    u_k = k_x * u + k_y * v + k_z * w;
    bv(gi, gj, gk, 1) = u - 2 * u_k * k_x;
    bv(gi, gj, gk, 2) = v - 2 * u_k * k_y;
    bv(gi, gj, gk, 3) = w - 2 * u_k * k_z;
    zone->vel(gi, gj, gk) = std::sqrt(bv(gi, gj, gk, 1) * bv(gi, gj, gk, 1) + bv(gi, gj, gk, 2) * bv(gi, gj, gk, 2) +
                                      bv(gi, gj, gk, 3) * bv(gi, gj, gk, 3));
    bv(gi, gj, gk, 4) = bv(ii, ij, ik, 4);
    bv(gi, gj, gk, 5) = bv(ii, ij, ik, 5);
    for (integer l = 0; l < param->n_scalar; ++l) {
      sv(gi, gj, gk, l) = sv(ii, ij, ik, l);
    }

    if constexpr (turb_method == TurbulenceMethod::RANS) {
      zone->mut(gi, gj, gk) = zone->mut(ii, ij, ik);
    }
  }
}

template<TurbulenceMethod turb_method>
__global__ void apply_outflow(DZone *zone, integer i_face, const DParameter *param) {
  const integer ngg = zone->ngg;
  integer dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  auto range_start = b.range_start, range_end = b.range_end;
  integer i = range_start[0] + (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  integer j = range_start[1] + (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  integer k = range_start[2] + (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  for (integer g = 1; g <= ngg; ++g) {
    const integer gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
    for (integer l = 0; l < 6; ++l) {
      bv(gi, gj, gk, l) = bv(i, j, k, l);
    }
    for (integer l = 0; l < param->n_scalar; ++l) {
      sv(gi, gj, gk, l) = sv(i, j, k, l);
    }
    if constexpr (turb_method == TurbulenceMethod::RANS) {
      zone->mut(gi, gj, gk) = zone->mut(i, j, k);
    }
  }
}

template<MixtureModel mix_model, TurbulenceMethod turb_method>
__global__ void apply_inflow(DZone *zone, Inflow *inflow, integer i_face, DParameter *param) {
  const integer ngg = zone->ngg;
  integer dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  auto range_start = b.range_start, range_end = b.range_end;
  integer i = range_start[0] + (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  integer j = range_start[1] + (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  integer k = range_start[2] + (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  const integer n_scalar = param->n_scalar;

  const real density = inflow->density;
  const real u = inflow->u;
  const real v = inflow->v;
  const real w = inflow->w;
  const auto *i_sv = inflow->sv;

  // Specify the boundary value as given.
  bv(i, j, k, 0) = density;
  bv(i, j, k, 1) = u;
  bv(i, j, k, 2) = v;
  bv(i, j, k, 3) = w;
  bv(i, j, k, 4) = inflow->pressure;
  bv(i, j, k, 5) = inflow->temperature;
  for (int l = 0; l < n_scalar; ++l) {
    sv(i, j, k, l) = i_sv[l];
  }
  if constexpr (turb_method == TurbulenceMethod::RANS) {
    zone->mut(i, j, k) = inflow->mut;
  }
  zone->vel(i, j, k) = inflow->velocity;

  for (integer g = 1; g <= ngg; g++) {
    const integer gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
    bv(gi, gj, gk, 0) = density;
    bv(gi, gj, gk, 1) = u;
    bv(gi, gj, gk, 2) = v;
    bv(gi, gj, gk, 3) = w;
    bv(gi, gj, gk, 4) = inflow->pressure;
    bv(gi, gj, gk, 5) = inflow->temperature;
    for (int l = 0; l < n_scalar; ++l) {
      sv(gi, gj, gk, l) = i_sv[l];
    }
    if constexpr (turb_method == TurbulenceMethod::RANS) {
      zone->mut(gi, gj, gk) = inflow->mut;
    }
  }
}

template<MixtureModel mix_model, TurbulenceMethod turb_method>
__global__ void apply_farfield(DZone *zone, FarField *farfield, integer i_face, DParameter *param) {
  const integer ngg = zone->ngg;
  integer dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  auto range_start = b.range_start, range_end = b.range_end;
  integer i = range_start[0] + (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  integer j = range_start[1] + (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  integer k = range_start[2] + (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;

  real nx{zone->metric(i, j, k)(b.face + 1, 1)},
      ny{zone->metric(i, j, k)(b.face + 1, 2)},
      nz{zone->metric(i, j, k)(b.face + 1, 3)};
  real grad_n_inv = b.direction / sqrt(nx * nx + ny * ny + nz * nz);
  nx *= grad_n_inv;
  ny *= grad_n_inv;
  nz *= grad_n_inv;
  const real u_b{bv(i, j, k, 1)}, v_b{bv(i, j, k, 2)}, w_b{bv(i, j, k, 3)};
  const real u_face{nx * u_b + ny * v_b + nz * w_b};

  // Interpolate the scalar values from internal nodes, which are used to compute gamma, after which, acoustic speed.
  const integer n_scalar = param->n_scalar, n_spec = param->n_spec;
  auto &sv = zone->sv;
  real gamma_b{gamma_air}, mw{mw_air};
  real sv_b[MAX_SPEC_NUMBER + 2], cp[MAX_SPEC_NUMBER];
  if constexpr (mix_model != MixtureModel::Air) {
    for (integer l = 0; l < n_scalar; ++l) {
      sv_b[l] = sv(i, j, k, l);
    }
    gamma_b = zone->gamma(i, j, k);
    real mw_inv{0};
    for (integer l = 0; l < n_spec; ++l) {
      mw_inv += sv_b[l] / param->mw[l];
    }
    mw = 1.0 / mw_inv;
  }
  const real p_b{bv(i, j, k, 4)}, rho_b{bv(i, j, k, 0)};
  const real a_b{sqrt(gamma_b * p_b / rho_b)};
  const real mach_b{u_face / a_b};

  if (mach_b <= -1) {
    // supersonic inflow
    const real density = farfield->density;
    const real u = farfield->u;
    const real v = farfield->v;
    const real w = farfield->w;
    const auto *i_sv = farfield->sv;

    // Specify the boundary value as given.
    bv(i, j, k, 0) = density;
    bv(i, j, k, 1) = u;
    bv(i, j, k, 2) = v;
    bv(i, j, k, 3) = w;
    bv(i, j, k, 4) = farfield->pressure;
    bv(i, j, k, 5) = farfield->temperature;
    for (int l = 0; l < n_scalar; ++l) {
      sv(i, j, k, l) = i_sv[l];
    }
    if constexpr (turb_method == TurbulenceMethod::RANS) {
      zone->mut(i, j, k) = farfield->mut;
    }
    zone->vel(i, j, k) = std::sqrt(u * u + v * v + w * w);


    for (integer g = 1; g <= ngg; g++) {
      const integer gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
      bv(gi, gj, gk, 0) = density;
      bv(gi, gj, gk, 1) = u;
      bv(gi, gj, gk, 2) = v;
      bv(gi, gj, gk, 3) = w;
      bv(gi, gj, gk, 4) = farfield->pressure;
      bv(gi, gj, gk, 5) = farfield->temperature;
      for (int l = 0; l < n_scalar; ++l) {
        sv(gi, gj, gk, l) = i_sv[l];
      }
      if constexpr (turb_method == TurbulenceMethod::RANS) {
        zone->mut(gi, gj, gk) = farfield->mut;
      }
    }
  } else if (mach_b >= 1) {
    // supersonic outflow
    for (integer g = 1; g <= ngg; ++g) {
      const integer gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
      for (integer l = 0; l < 6; ++l) {
        bv(gi, gj, gk, l) = bv(i, j, k, l);
      }
      for (integer l = 0; l < n_scalar; ++l) {
        sv(gi, gj, gk, l) = sv(i, j, k, l);
      }
      if constexpr (turb_method == TurbulenceMethod::RANS) {
        zone->mut(gi, gj, gk) = zone->mut(i, j, k);
      }
    }
  } else {
    // subsonic inflow and outflow

    // The positive riemann invariant is in the same direction of the boundary normal, which points to the outside of the computational domain.
    // Thus, it is computed from the internal nodes.
    const real riemann_pos{u_face + 2 * a_b / (gamma_b - 1)};
    const real u_inf{farfield->u * nx + farfield->v * ny + farfield->w * nz};
    const real riemann_neg{u_inf - 2 * farfield->acoustic_speed / (farfield->specific_heat_ratio - 1)};

    real s_b, density, pressure, temperature, u, v, w, mut;
    const real Un{0.5 * (riemann_pos + riemann_neg)};
    if constexpr (mix_model == MixtureModel::Air) {
      const real c_b{0.25 * (gamma_air - 1) * (riemann_pos - riemann_neg)};
      if (mach_b <= 0) {
        // inflow
        s_b = farfield->entropy;
        u = farfield->u + (Un - u_inf) * nx;
        v = farfield->v + (Un - u_inf) * ny;
        w = farfield->w + (Un - u_inf) * nz;
        for (int l = 0; l < n_scalar; ++l) {
          sv_b[l] = farfield->sv[l];
        }
        mut = farfield->mut;
      } else {
        // outflow
        s_b = p_b / pow(rho_b, gamma_air);
        u = u_b + (Un - u_face) * nx;
        v = v_b + (Un - u_face) * ny;
        w = w_b + (Un - u_face) * nz;
        mut = zone->mut(i, j, k);
      }
      density = pow(c_b * c_b / (gamma_air * s_b), 1 / (gamma_air - 1));
      pressure = density * c_b * c_b / gamma_air;
      temperature = pressure * mw_air / (density * R_u);
    } else {
      // Mixture
      if (mach_b < 0) {
        // inflow
        u = farfield->u + (Un - u_inf) * nx;
        v = farfield->v + (Un - u_inf) * ny;
        w = farfield->w + (Un - u_inf) * nz;
        for (integer l = 0; l < n_scalar; ++l) {
          sv_b[l] = farfield->sv[l];
        }
        mw = farfield->mw;
        mut = farfield->mut;
      } else {
        // outflow
        u = u_b + (Un - u_face) * nx;
        v = v_b + (Un - u_face) * ny;
        w = w_b + (Un - u_face) * nz;
        // When this is outflow, the sv_b should be interpolated from internal points, which has been computed above
        mut = zone->mut(i, j, k);
      }
      real gamma{gamma_air}, err{1}, gamma_last{gamma_air};
      while (err > 1e-4) {
        gamma_last = gamma;
        const real c_b{0.25 * (gamma - 1) * (riemann_pos - riemann_neg)};
        if (mach_b <= 0) {
          // inflow
          s_b = farfield->entropy;
        } else {
          // outflow
          s_b = p_b / pow(rho_b, gamma);
        }
        density = pow(c_b * c_b / (gamma * s_b), 1 / (gamma - 1));
        pressure = density * c_b * c_b / gamma;
        temperature = pressure * mw / (density * R_u);
        compute_cp(temperature, cp, param);
        real cp_tot{0};
        for (integer l = 0; l < n_spec; ++l) {
          cp_tot += cp[l] * sv_b[l];
        }
        gamma = cp_tot / (cp_tot - R_u / mw);
        err = abs(1 - gamma / gamma_last);
      }
    }

    // Specify the boundary value as given.
    bv(i, j, k, 0) = density;
    bv(i, j, k, 1) = u;
    bv(i, j, k, 2) = v;
    bv(i, j, k, 3) = w;
    bv(i, j, k, 4) = pressure;
    bv(i, j, k, 5) = temperature;
    for (int l = 0; l < n_scalar; ++l) {
      sv(i, j, k, l) = sv_b[l];
    }
    if constexpr (turb_method == TurbulenceMethod::RANS) {
      zone->mut(i, j, k) = mut;
    }
    zone->vel(i, j, k) = std::sqrt(u * u + v * v + w * w);

    for (integer g = 1; g <= ngg; g++) {
      const integer gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
      bv(gi, gj, gk, 0) = density;
      bv(gi, gj, gk, 1) = u;
      bv(gi, gj, gk, 2) = v;
      bv(gi, gj, gk, 3) = w;
      bv(gi, gj, gk, 4) = pressure;
      bv(gi, gj, gk, 5) = temperature;
      for (int l = 0; l < n_scalar; ++l) {
        sv(gi, gj, gk, l) = sv_b[l];
      }
      if constexpr (turb_method == TurbulenceMethod::RANS) {
        zone->mut(gi, gj, gk) = mut;
      }
    }
  }

}

template<MixtureModel mix_model, TurbulenceMethod turb_method>
__global__ void apply_wall(DZone *zone, Wall *wall, DParameter *param, integer i_face) {
  const integer ngg = zone->ngg;
  integer dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  auto range_start = b.range_start, range_end = b.range_end;
  integer i = range_start[0] + (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  integer j = range_start[1] + (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  integer k = range_start[2] + (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;
  const integer n_spec = param->n_spec;

  real t_wall{wall->temperature};

  const integer idx[]{i - dir[0], j - dir[1], k - dir[2]};
  if (wall->thermal_type == Wall::ThermalType::adiabatic) {
    t_wall = bv(idx[0], idx[1], idx[2], 5);
  }
  const real p{bv(idx[0], idx[1], idx[2], 4)};

  real mw{cfd::mw_air};
  if constexpr (mix_model != MixtureModel::Air) {
    // Mixture
    const auto mwk = param->mw;
    mw = 0;
    for (integer l = 0; l < n_spec; ++l) {
      sv(i, j, k, l) = sv(idx[0], idx[1], idx[2], l);
      mw += sv(i, j, k, l) / mwk[l];
    }
    mw = 1 / mw;
  }

  const real rho_wall = p * mw / (t_wall * cfd::R_u);
  bv(i, j, k, 0) = rho_wall;
  bv(i, j, k, 1) = 0;
  bv(i, j, k, 2) = 0;
  bv(i, j, k, 3) = 0;
  bv(i, j, k, 4) = p;
  bv(i, j, k, 5) = t_wall;
  zone->vel(i, j, k) = 0;

  // turbulent boundary condition
  if constexpr (turb_method == TurbulenceMethod::RANS) {
    if (param->rans_model == 2) {
      // SST
      real mu_wall{0};
      if constexpr (mix_model != MixtureModel::Air) {
        mu_wall = compute_viscosity(i, j, k, t_wall, mw, param, zone);
      } else {
        mu_wall = Sutherland(t_wall);
      }
      const real dy = zone->wall_distance(idx[0], idx[1], idx[2]);
      sv(i, j, k, n_spec) = 0;
      if (dy > 1e-25) {
        sv(i, j, k, n_spec + 1) = 60 * mu_wall / (rho_wall * SST::beta_1 * dy * dy);
      } else {
        sv(i, j, k, n_spec + 1) = sv(idx[0], idx[1], idx[2], n_spec + 1);
      }
    }
  }

  if constexpr (mix_model == MixtureModel::FL) {
    // Flamelet model
    const integer i_fl{param->i_fl};
    sv(i, j, k, i_fl) = sv(idx[0], idx[1], idx[2], i_fl);
    sv(i, j, k, i_fl + 1) = sv(idx[0], idx[1], idx[2], i_fl + 1);
  }

  for (int g = 1; g <= ngg; ++g) {
    const integer i_in[]{i - g * dir[0], j - g * dir[1], k - g * dir[2]};
    const integer i_gh[]{i + g * dir[0], j + g * dir[1], k + g * dir[2]};

    const real u_i{bv(i_in[0], i_in[1], i_in[2], 1)};
    const real v_i{bv(i_in[0], i_in[1], i_in[2], 2)};
    const real w_i{bv(i_in[0], i_in[1], i_in[2], 3)};
    const real p_i{bv(i_in[0], i_in[1], i_in[2], 4)};
    const real t_i{bv(i_in[0], i_in[1], i_in[2], 5)};

    double t_g{t_i};
    if (wall->thermal_type == Wall::ThermalType::isothermal) {
      t_g = 2 * t_wall - t_i;  // 0.5*(t_i+t_g)=t_w
      if (t_g <= 0.1 * t_wall) { // improve stability
        t_g = t_wall;
      }
    }

    if constexpr (mix_model != MixtureModel::Air) {
      const auto mwk = param->mw;
      mw = 0;
      for (integer l = 0; l < param->n_spec; ++l) {
        sv(i_gh[0], i_gh[1], i_gh[2], l) = sv(i_in[0], i_in[1], i_in[2], l);
        mw += sv(i_gh[0], i_gh[1], i_gh[2], l) / mwk[l];
      }
      mw = 1 / mw;
    }

    const real rho_g{p_i * mw / (t_g * cfd::R_u)};
    bv(i_gh[0], i_gh[1], i_gh[2], 0) = rho_g;
    bv(i_gh[0], i_gh[1], i_gh[2], 1) = -u_i;
    bv(i_gh[0], i_gh[1], i_gh[2], 2) = -v_i;
    bv(i_gh[0], i_gh[1], i_gh[2], 3) = -w_i;
    bv(i_gh[0], i_gh[1], i_gh[2], 4) = p_i;
    bv(i_gh[0], i_gh[1], i_gh[2], 5) = t_g;

    // turbulent boundary condition
    if constexpr (turb_method == TurbulenceMethod::RANS) {
      if (param->rans_model == 2) {
        // SST
        sv(i_gh[0], i_gh[1], i_gh[2], n_spec) = 0;
        sv(i_gh[0], i_gh[1], i_gh[2], n_spec + 1) = sv(i, j, k, n_spec + 1);
        zone->mut(i_gh[0], i_gh[1], i_gh[2]) = 0;
      }
    }

    if constexpr (mix_model == MixtureModel::FL) {
      sv(i_gh[0], i_gh[1], i_gh[2], param->i_fl) = sv(i_in[0], i_in[1], i_in[2], param->i_fl);
      sv(i_gh[0], i_gh[1], i_gh[2], param->i_fl + 1) = sv(i_in[0], i_in[1], i_in[2], param->i_fl + 1);
    }
  }
}

template<MixtureModel mix_model, TurbulenceMethod turb_method>
__global__ void apply_subsonic_inflow(DZone *zone, SubsonicInflow *inflow, DParameter *param, integer i_face) {
  const integer ngg = zone->ngg;
  integer dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  auto range_start = b.range_start, range_end = b.range_end;
  integer i = range_start[0] + (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  integer j = range_start[1] + (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  integer k = range_start[2] + (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  // Compute the normal direction of the face. The direction is from the inside to the outside of the computational domain.
  real nx{zone->metric(i, j, k)(b.face + 1, 1)},
      ny{zone->metric(i, j, k)(b.face + 1, 2)},
      nz{zone->metric(i, j, k)(b.face + 1, 3)};
  real grad_n_inv = b.direction / sqrt(nx * nx + ny * ny + nz * nz);
  nx *= grad_n_inv;
  ny *= grad_n_inv;
  nz *= grad_n_inv;
  const real u_face{nx * bv(i, j, k, 1) + ny * bv(i, j, k, 2) + nz * bv(i, j, k, 3)};
  // compute the negative Riemann invariant with computed boundary value.
  const real acoustic_speed{sqrt(gamma_air * bv(i, j, k, 4) / bv(i, j, k, 0))};
  const real riemann_neg{abs(u_face) - 2 * acoustic_speed / (gamma_air - 1)};
  // compute the total enthalpy of the inflow.
  const real hti{bv(i, j, k, 4) / bv(i, j, k, 0) * gamma_air / (gamma_air - 1) +
                 0.5 * (bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) +
                        bv(i, j, k, 3) * bv(i, j, k, 3))};
  constexpr real qa{1 + 2.0 / (gamma_air - 1)};
  const real qb{2 * riemann_neg};
  const real qc{(gamma_air - 1) * (0.5 * riemann_neg * riemann_neg - hti)};
  const real delta{qb * qb - 4 * qa * qc};
  real a_new{acoustic_speed};
  if (delta >= 0) {
    real a_plus{(-qb + sqrt(delta)) / (2 * qa)};
    real a_minus{(-qb - sqrt(delta)) / (2 * qa)};
    a_new = a_plus > a_minus ? a_plus : a_minus;
  }

  const real u_new{riemann_neg + 2 * a_new / (gamma_air - 1)};
  const real mach{u_new / a_new};
  const real pressure{
      inflow->total_pressure * pow(1 + 0.5 * (gamma_air - 1) * mach * mach, -gamma_air / (gamma_air - 1))};
  const real temperature{
      inflow->total_temperature * pow(pressure / inflow->total_pressure, (gamma_air - 1) / gamma_air)};
  const real density{pressure * mw_air / (temperature * cfd::R_u)};

  // assign values for ghost cells
  for (integer g = 1; g <= ngg; g++) {
    const integer gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
    bv(gi, gj, gk, 0) = density;
    bv(gi, gj, gk, 1) = u_new * inflow->u;
    bv(gi, gj, gk, 2) = u_new * inflow->v;
    bv(gi, gj, gk, 3) = u_new * inflow->w;
    bv(gi, gj, gk, 4) = pressure;
    bv(gi, gj, gk, 5) = temperature;

    if constexpr (turb_method == TurbulenceMethod::RANS) {
      const real u_bar{bv(gi, gj, gk, 1) * nx + bv(gi, gj, gk, 2) * ny + bv(gi, gj, gk, 3) * nz};
      const integer n_scalar = param->n_scalar;
      if (u_bar > 0) {
        // The normal velocity points out of the domain, which means the value should be acquired from internal nodes.
        for (int l = 0; l < n_scalar; ++l) {
          sv(gi, gj, gk, l) = sv(i, j, k, l);
        }
      } else {
        // The normal velocity points into the domain, which means the value should be acquired from the boundary.
        for (int l = 0; l < n_scalar; ++l) {
          sv(gi, gj, gk, l) = inflow->sv[l];
        }
      }

      // In CFL3D, only the first ghost layer is assigned with the value on the boundary, and the rest are assigned with 0.
      zone->mut(gi, gj, gk) = zone->mut(i, j, k);
    }
  }
}

template<MixtureModel mix_model, TurbulenceMethod turb_method>
__global__ void apply_back_pressure(DZone *zone, BackPressure *backPressure, DParameter *param, integer i_face) {
  const integer ngg = zone->ngg;
  integer dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  auto range_start = b.range_start, range_end = b.range_end;
  integer i = range_start[0] + (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  integer j = range_start[1] + (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  integer k = range_start[2] + (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  for (integer g = 1; g <= ngg; ++g) {
    const integer gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
    for (integer l = 0; l < 4; ++l) {
      bv(gi, gj, gk, l) = bv(i, j, k, l);
    }
    bv(gi, gj, gk, 4) = backPressure->pressure;
    // This should be modified later, as p is specified, temperature is extrapolated, the density should be acquired from equation of state instead of extrapolation.
    bv(gi, gj, gk, 5) = bv(i, j, k, 5);
    for (integer l = 0; l < param->n_scalar; ++l) {
      sv(gi, gj, gk, l) = sv(i, j, k, l);
    }
    if constexpr (turb_method == TurbulenceMethod::RANS) {
      zone->mut(gi, gj, gk) = zone->mut(i, j, k);
    }
  }
}

template<MixtureModel mix_model, TurbulenceMethod turb_method>
void DBoundCond::apply_boundary_conditions(const Block &block, Field &field, DParameter *param) const {
  // Boundary conditions are applied in the order of priority, which with higher priority is applied later.
  // Finally, the communication between faces will be carried out after these bc applied
  // Priority: (-1 - inner faces >) 2-wall > 3-symmetry > 5-inflow = 7-subsonic inflow > 6-outflow = 9-back pressure > 4-farfield

  // 4-farfield
  for (size_t l = 0; l < n_farfield; ++l) {
    const auto nb = farfield_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = farfield_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &h_f = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = h_f.range_end[j] - h_f.range_start[j] + 1;
        tpb[j] = n_point <= (2 * ngg + 1) ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_farfield<mix_model, turb_method> <<<BPG, TPB>>>(field.d_ptr, &farfield[l], i_face, param);
    }
  }

  // 6-outflow
  for (size_t l = 0; l < n_outflow; l++) {
    const auto nb = outflow_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = outflow_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &h_f = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = h_f.range_end[j] - h_f.range_start[j] + 1;
        tpb[j] = n_point <= (2 * ngg + 1) ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_outflow<turb_method> <<<BPG, TPB>>>(field.d_ptr, i_face, param);
    }
  }
  for (size_t l = 0; l < n_back_pressure; l++) {
    const auto nb = back_pressure_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = back_pressure_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &h_f = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = h_f.range_end[j] - h_f.range_start[j] + 1;
        tpb[j] = n_point <= (2 * ngg + 1) ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_back_pressure<mix_model, turb_method> <<<BPG, TPB>>>(field.d_ptr, &back_pressure[l], param, i_face);
    }
  }

  // 5-inflow
  for (size_t l = 0; l < n_inflow; l++) {
    const auto nb = inflow_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = inflow_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &hf = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = hf.range_end[j] - hf.range_start[j] + 1;
        tpb[j] = n_point <= (2 * ngg + 1) ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_inflow<mix_model, turb_method> <<<BPG, TPB>>>(field.d_ptr, &inflow[l], i_face, param);
    }
  }
  // 7 - subsonic inflow
  for (size_t l = 0; l < n_subsonic_inflow; l++) {
    const auto nb = subsonic_inflow_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = subsonic_inflow_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &hf = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = hf.range_end[j] - hf.range_start[j] + 1;
        tpb[j] = n_point <= (2 * ngg + 1) ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_subsonic_inflow<mix_model, turb_method> <<<BPG, TPB>>>(field.d_ptr, &subsonic_inflow[l], param, i_face);
    }
  }

  // 3-symmetry
  for (size_t l = 0; l < n_symmetry; l++) {
    const auto nb = symmetry_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = symmetry_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &h_f = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = h_f.range_end[j] - h_f.range_start[j] + 1;
        tpb[j] = n_point <= (2 * ngg + 1) ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_symmetry<mix_model, turb_method> <<<BPG, TPB>>>(field.d_ptr, i_face, param);
    }
  }

  // 2 - wall
  for (size_t l = 0; l < n_wall; l++) {
    const auto nb = wall_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = wall_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &hf = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = hf.range_end[j] - hf.range_start[j] + 1;
        tpb[j] = n_point <= (2 * ngg + 1) ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_wall<mix_model, turb_method><<<BPG, TPB>>>(field.d_ptr, &wall[l], param, i_face);
    }
  }
}

} // cfd

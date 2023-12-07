#include "AWENO.cuh"
#include "Parallel.h"
#include "Field.h"
#include "DParameter.cuh"
#include "HLLC.cuh"
#include "Constants.h"

namespace cfd {

template<MixtureModel mix_model>
__device__ void
AWENO_interpolation(const real *pv, real *pv_l, real *pv_r, integer idx_shared, integer n_var, const real *metric,
                    DParameter *param) {
  // For mixture, this is not implemented.
}

template<>
__device__ void
AWENO_interpolation<MixtureModel::Air>(const real *cv, real *pv_l, real *pv_r, integer idx_shared, integer n_var,
                                       const real *metric, DParameter *param) {
  // The first n_var in the cv array is conservative vars, followed by p and T.
  const real *cvl{&cv[idx_shared * (n_var + 2)]};
  const real *cvr{&cv[(idx_shared + 1) * (n_var + 2)]};
//  real sv_average[5]; // 5 for scalar variables, we think the max number of scalars for air simulation should not exceed 5.
  // First, compute the Roe average of the half-point variables.
  const real rlc{sqrt(cvl[0]) / (sqrt(cvl[0]) + sqrt(cvr[0]))};
  const real rrc{sqrt(cvr[0]) / (sqrt(cvl[0]) + sqrt(cvr[0]))};
  const real um{rlc * cvl[1] / cvl[0] + rrc * cvr[1] / cvr[0]};
  const real vm{rlc * cvl[2] / cvl[0] + rrc * cvr[2] / cvr[0]};
  const real wm{rlc * cvl[3] / cvl[0] + rrc * cvr[3] / cvr[0]};
  const real ekm{0.5 * (um * um + vm * vm + wm * wm)};
  constexpr real gm1{gamma_air - 1};
  const real hl{(cvl[4] + cvl[n_var]) / cvl[0]};
  const real hr{(cvr[4] + cvr[n_var]) / cvr[0]};
  const real hm{rlc * hl + rrc * hr};
  const real cm2{gm1 * (hm - ekm)};
  const real cm{sqrt(cm2)};
//  for (auto l = 5; l < n_var; ++l) {
//    // Scalars
//    sv_average[l - 5] = rlc * cvl[l] / cvl[0] + rrc * cvr[l] / cvr[0];
//  }

  // Next, we compute the left characteristic matrix at i+1/2.
  real kx{0.5 * (metric[idx_shared * 3] + metric[(idx_shared + 1) * 3])};
  real ky{0.5 * (metric[idx_shared * 3 + 1] + metric[(idx_shared + 1) * 3 + 1])};
  real kz{0.5 * (metric[idx_shared * 3 + 2] + metric[(idx_shared + 1) * 3 + 2])};
  const real gradK{sqrt(kx * kx + ky * ky + kz * kz)};
  kx /= gradK;
  ky /= gradK;
  kz /= gradK;
  const real Uk_bar{kx * um + ky * vm + kz * wm};
  const real alpha{gm1 * ekm};
  // The matrix we consider here does not contain the turbulent variables, such as tke and omega.
  // The method which contains turbulent variables is not implemented yet.
  gxl::Matrix<real, 5, 5> LR;
  LR(0, 0) = (alpha + Uk_bar * cm) / cm2 * 0.5;
  LR(0, 1) = -(gm1 * um + kx * cm) / cm2 * 0.5;
  LR(0, 2) = -(gm1 * vm + ky * cm) / cm2 * 0.5;
  LR(0, 3) = -(gm1 * wm + kz * cm) / cm2 * 0.5;
  LR(0, 4) = gm1 / cm2 * 0.5;
  LR(1, 0) = kx * (1 - alpha / cm2) - (kz * vm - ky * wm) / cm;
  LR(1, 1) = kx * gm1 * um / cm2;
  LR(1, 2) = (kx * gm1 * vm + kz * cm) / cm2;
  LR(1, 3) = (kx * gm1 * wm - ky * cm) / cm2;
  LR(1, 4) = -kx * gm1 / cm2;
  LR(2, 0) = ky * (1 - alpha / cm2) - (kx * wm - kz * um) / cm;
  LR(2, 1) = (ky * gm1 * um - kz * cm) / cm2;
  LR(2, 2) = ky * gm1 * vm / cm2;
  LR(2, 3) = (ky * gm1 * wm + kx * cm) / cm2;
  LR(2, 4) = -ky * gm1 / cm2;
  LR(3, 0) = kz * (1 - alpha / cm2) - (ky * um - kx * vm) / cm;
  LR(3, 1) = (kz * gm1 * um + ky * cm) / cm2;
  LR(3, 2) = (kz * gm1 * vm - kx * cm) / cm2;
  LR(3, 3) = kz * gm1 * wm / cm2;
  LR(3, 4) = -kz * gm1 / cm2;
  LR(4, 0) = (alpha - Uk_bar * cm) / cm2 * 0.5;
  LR(4, 1) = -(gm1 * um - kx * cm) / cm2 * 0.5;
  LR(4, 2) = -(gm1 * vm - ky * cm) / cm2 * 0.5;
  LR(4, 3) = -(gm1 * wm - kz * cm) / cm2 * 0.5;
  LR(4, 4) = gm1 / cm2 * 0.5;

  // Interpolate the characteristic variable with WENO
  real v_plus[5], v_minus[5];
  memset(v_plus, 0, 5 * sizeof(real));
  memset(v_minus, 0, 5 * sizeof(real));
  if (param->reconstruction == 4) {
    // WENO 5
    for (integer l = 0; l < 5; ++l) {
      // We reconstruct each characteristic variable to reduce the memory to be used.
      // WENO5(L.data(), cv, n_var, idx_shared, l);
      auto v2 = WENO5(LR.data(), cv, 5, idx_shared, l);
      v_minus[l] = v2.x;
      v_plus[l] = v2.y;
    }
  }

  // Compute the right characteristic matrix
  LR(0, 0) = 1.0;
  LR(0, 1) = kx;
  LR(0, 2) = ky;
  LR(0, 3) = kz;
  LR(0, 4) = 1.0;
  LR(1, 0) = um - kx * cm;
  LR(1, 1) = kx * um;
  LR(1, 2) = ky * um - kz * cm;
  LR(1, 3) = kz * um + ky * cm;
  LR(1, 4) = um + kx * cm;
  LR(2, 0) = vm - ky * cm;
  LR(2, 1) = kx * vm + kz * cm;
  LR(2, 2) = ky * vm;
  LR(2, 3) = kz * vm - kx * cm;
  LR(2, 4) = vm + ky * cm;
  LR(3, 0) = wm - kz * cm;
  LR(3, 1) = kx * wm - ky * cm;
  LR(3, 2) = ky * wm + kx * cm;
  LR(3, 3) = kz * wm;
  LR(3, 4) = wm + kz * cm;
  LR(4, 0) = hm - Uk_bar * cm;
  LR(4, 1) = kx * alpha / gm1 + (kz * vm - ky * wm) * cm;
  LR(4, 2) = ky * alpha / gm1 + (kx * wm - kz * um) * cm;
  LR(4, 3) = kz * alpha / gm1 + (ky * um - kx * vm) * cm;
  LR(4, 4) = hm + Uk_bar * cm;

  // Project the "v+" and "v-" back to physical space
  real ql[5], qr[5];
  for (integer m = 0; m < 5; ++m) {
    ql[m] = 0;
    qr[m] = 0;
    for (integer n = 0; n < 5; ++n) {
      ql[m] += LR(m, n) * v_minus[n];
      qr[m] += LR(m, n) * v_plus[n];
    }
  }

  // Compute the basic variables from conservative variables
  pv_l[0] = ql[0];
  pv_l[1] = ql[1] / ql[0];
  pv_l[2] = ql[2] / ql[0];
  pv_l[3] = ql[3] / ql[0];
  pv_l[4] = gm1 * (ql[4] - 0.5 * (pv_l[1] * pv_l[1] + pv_l[2] * pv_l[2] + pv_l[3] * pv_l[3]) * ql[0]);
  pv_l[n_var] = ql[4];

  pv_r[0] = qr[0];
  pv_r[1] = qr[1] / qr[0];
  pv_r[2] = qr[2] / qr[0];
  pv_r[3] = qr[3] / qr[0];
  pv_r[4] = gm1 * (qr[4] - 0.5 * (pv_r[1] * pv_r[1] + pv_r[2] * pv_r[2] + pv_r[3] * pv_r[3]) * qr[0]);
  pv_r[n_var] = qr[4];
}

__device__ double2 WENO5(const real *L, const real *cv, integer n_var, integer i_shared, integer l_row) {
  // The indices for them are i-2 to i+3.
  // 0 - i-2, 1 - i-1, 2 - i, 3 - i+1, 4 - i+2, 5 - i+3

  // Project to characteristic space
  real v[6];
  memset(v, 0, 6 * sizeof(real));
  for (integer i = -2; i < 4; ++i) {
    const integer idx{i_shared + i};
    const real *U{&cv[idx * (n_var + 2)]};
    for (integer j = 0; j < n_var; ++j) {
      v[i + 2] += L[l_row * n_var + j] * U[j];
    }
  }

  // The coefficients here can be acquired from "Liu, H (2017) A numerical study of the performance of
  // alternative weighted ENO methods based on various numerical fluxes for conservation law.
  // Applied Mathematics and Computation, 296, 182–197. doi:10.1016/j.amc.2016.10.023."

  // Reconstruct the "v-" with v[-2:2]
  real v0{0.125 * (3 * v[0] - 10 * v[1] + 15 * v[2])};
  real v1{0.125 * (-v[1] + 6 * v[2] + 3 * v[3])};
  real v2{0.125 * (3 * v[2] + 6 * v[3] - v[4])};
  constexpr real thirteen12th{13.0 / 12};
  real beta0 = thirteen12th * (v[0] + v[2] - 2 * v[1]) * (v[0] + v[2] - 2 * v[1]) +
               0.25 * (v[0] + 3 * v[2] - 4 * v[1]) * (v[0] + 3 * v[2] - 4 * v[1]);
  real beta1 = thirteen12th * (v[1] + v[3] - 2 * v[2]) * (v[1] + v[3] - 2 * v[2]) +
               0.25 * (v[3] - v[1]) * (v[3] - v[1]);
  real beta2 = thirteen12th * (v[2] + v[4] - 2 * v[3]) * (v[2] + v[4] - 2 * v[3]) +
               0.25 * (3 * v[2] + v[4] - 4 * v[3]) * (3 * v[2] + v[4] - 4 * v[3]);
  real tau5sqr{(beta0 - beta2) * (beta0 - beta2)};
  constexpr real eps{1e-6};
  constexpr real oneDiv16{1.0 / 16}, fiveDiv8{5.0 / 8}, fiveDiv16{5.0 / 16};
  real a0{oneDiv16 + oneDiv16 * tau5sqr / ((eps + beta0) * (eps + beta0))};
  real a1{fiveDiv8 + fiveDiv8 * tau5sqr / ((eps + beta1) * (eps + beta1))};
  real a2{fiveDiv16 + fiveDiv16 * tau5sqr / ((eps + beta2) * (eps + beta2))};
  const real v_minus{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

  // Reconstruct the "v+" with v[-1:3]
  v0 = v1;
  v1 = v2;
  v2 = 0.125 * (15 * v[3] - 10 * v[4] + 3 * v[5]);
  beta0 = thirteen12th * (v[1] + v[3] - 2 * v[2]) * (v[1] + v[3] - 2 * v[2]) +
          0.25 * (v[1] + 3 * v[3] - 4 * v[2]) * (v[1] + 3 * v[3] - 4 * v[2]);
  beta1 = thirteen12th * (v[2] + v[4] - 2 * v[3]) * (v[2] + v[4] - 2 * v[3]) +
          0.25 * (v[4] - v[2]) * (v[4] - v[2]);
  beta2 = thirteen12th * (v[3] + v[5] - 2 * v[4]) * (v[3] + v[5] - 2 * v[4]) +
          0.25 * (3 * v[3] + v[5] - 4 * v[4]) * (3 * v[3] + v[5] - 4 * v[4]);
  tau5sqr = (beta0 - beta2) * (beta0 - beta2);
  a0 = fiveDiv16 + fiveDiv16 * tau5sqr / ((eps + beta0) * (eps + beta0));
  a1 = fiveDiv8 + fiveDiv8 * tau5sqr / ((eps + beta1) * (eps + beta1));
  a2 = oneDiv16 + oneDiv16 * tau5sqr / ((eps + beta2) * (eps + beta2));
  const real v_plus{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

  return double2{v_minus, v_plus};
}

template<MixtureModel mix_model>
__global__ void CDSPart1D(cfd::DZone *zone, integer direction, integer max_extent, DParameter *param) {
  printf("Not implemented.\n");
}

template<>
__global__ void
CDSPart1D<MixtureModel::Air>(cfd::DZone *zone, integer direction, integer max_extent, DParameter *param) {
  integer labels[3]{0, 0, 0};
  labels[direction] = 1;
  const auto tid = (integer) (threadIdx.x * labels[0] + threadIdx.y * labels[1] + threadIdx.z * labels[2]);
  const auto block_dim = (integer) (blockDim.x * blockDim.y * blockDim.z);
  const auto ngg{zone->ngg};
//  const integer n_point = block_dim + 2 * ngg - 1;

  integer idx[3];
  idx[0] = (integer) ((blockDim.x - 2 * ngg * labels[0]) * blockIdx.x + threadIdx.x);
  idx[1] = (integer) ((blockDim.y - 2 * ngg * labels[1]) * blockIdx.y + threadIdx.y);
  idx[2] = (integer) ((blockDim.z - 2 * ngg * labels[2]) * blockIdx.z + threadIdx.z);
  idx[direction] -= ngg;
  if (idx[direction] >= max_extent - ngg) return;

  extern __shared__ real f[];
  const auto n_var{param->n_var};

  // Compute the flux with the corresponding cv and bv.
  const auto &cv = zone->cv;
  const auto &bv = zone->bv;
  const auto &metric = zone->metric(idx[0], idx[1], idx[2]);
  const auto jac = zone->jac(idx[0], idx[1], idx[2]);

  const real kx{metric(direction + 1, 1)}, ky{metric(direction + 1, 2)}, kz{metric(direction + 1, 3)};
  const real u{bv(idx[0], idx[1], idx[2], 1)}, v{bv(idx[0], idx[1], idx[2], 2)}, w{bv(idx[0], idx[1], idx[2], 3)};
  const real Uk{kx * u + ky * v + kz * w};
  const real rhoUk{bv(idx[0], idx[1], idx[2], 0) * Uk};
  const real p{bv(idx[0], idx[1], idx[2], 4)};

  f[tid * n_var] = rhoUk * jac;
  f[tid * n_var + 1] = jac * (rhoUk * u + kx * p);
  f[tid * n_var + 2] = jac * (rhoUk * v + ky * p);
  f[tid * n_var + 3] = jac * (rhoUk * w + kz * p);
  f[tid * n_var + 4] = jac * Uk * (cv(idx[0], idx[1], idx[2], 4) + p);

  __syncthreads();

  if (tid < ngg || tid >= block_dim - ngg || idx[direction] >= max_extent - 2 * ngg)
    return;

  // The central part compute the flux contributed by these fluxes.
  constexpr real c1{19.0 / 3840}, c2{-13.0 / 320}, c3{17.0 / 256};
  for (integer l = 0; l < n_var; ++l) {
    zone->dq(idx[0], idx[1], idx[2], l) -=
        c1 * (f[(tid + 3) * n_var + l] - f[(tid - 3) * n_var + l]) +
        c2 * (f[(tid + 2) * n_var + l] - f[(tid - 2) * n_var + l]) +
        c3 * (f[(tid + 1) * n_var + l] - f[(tid - 1) * n_var + l]);
  }
}

template __device__ void
AWENO_interpolation<MixtureModel::Mixture>(const real *cv, real *pv_l, real *pv_r, integer idx_shared, integer n_var,
                                           const real *metric, DParameter *param);

template __device__ void
AWENO_interpolation<MixtureModel::FR>(const real *cv, real *pv_l, real *pv_r, integer idx_shared, integer n_var,
                                      const real *metric, DParameter *param);

template __device__ void
AWENO_interpolation<MixtureModel::FL>(const real *cv, real *pv_l, real *pv_r, integer idx_shared, integer n_var,
                                      const real *metric, DParameter *param);

template __device__ void
AWENO_interpolation<MixtureModel::MixtureFraction>(const real *cv, real *pv_l, real *pv_r, integer idx_shared,
                                                   integer n_var, const real *metric, DParameter *param);

template __global__ void
CDSPart1D<MixtureModel::Mixture>(cfd::DZone *zone, integer direction, integer max_extent, DParameter *param);

template __global__ void
CDSPart1D<MixtureModel::FR>(cfd::DZone *zone, integer direction, integer max_extent, DParameter *param);

template __global__ void
CDSPart1D<MixtureModel::FL>(cfd::DZone *zone, integer direction, integer max_extent, DParameter *param);

template __global__ void
CDSPart1D<MixtureModel::MixtureFraction>(cfd::DZone *zone, integer direction, integer max_extent, DParameter *param);

}
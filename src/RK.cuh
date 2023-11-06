#pragma once

#include "Driver.cuh"
#include "FieldOperation.cuh"

namespace cfd {

namespace SSPRK3 {
__device__ constexpr real a[3]{1.0, 0.75, 1.0 / 3.0};
__device__ constexpr real b[3]{0.0, 0.25, 2.0 / 3.0};
__device__ constexpr real c[3]{1.0, 0.25, 2.0 / 3.0};
}

template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void update_cv_and_bv_rk(cfd::DZone *zone, DParameter *param, real dt, integer rk);

template<MixtureModel mix_model, class turb>
void RK3_bv(Driver<mix_model, turb> &driver) {
  if (driver.myid == 0) {
    printf("Unsteady flow simulation with 3rd order Runge-Kutta scheme for time advancing.\n");
  }

  dim3 tpb{8, 8, 4};
  auto &mesh{driver.mesh};
  const integer n_block{mesh.n_block};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  dim3 *bpg = new dim3[n_block];
  for (integer b = 0; b < n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    bpg[b] = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
  }

  std::vector<cfd::Field> &field{driver.field};
  const integer ng_1 = 2 * mesh[0].ngg - 1;
  DParameter *param{driver.param};
  for (auto b = 0; b < n_block; ++b) {
    // Store the initial value of the flow field
    store_last_step<<<bpg[b], tpb>>>(field[b].d_ptr);
    // Compute the conservative variables from basic variables
    // In unsteady simulations, because of the upwind high-order method to be used;
    // we need the conservative variables.
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
    compute_cv_from_bv<mix_model, turb><<<BPG, tpb>>>(field[b].d_ptr, param);
  }

  auto &parameter{driver.parameter};
  IOManager<mix_model, turb> ioManager(driver.myid, mesh, field, parameter, driver.spec, 0);
  TimeSeriesIOManager<mix_model, turb> timeSeriesIOManager(driver.myid, mesh, field, parameter, driver.spec, 0);

  integer step{parameter.get_int("step")};
  integer total_step{parameter.get_int("total_step") + step};
  const integer output_screen = parameter.get_int("output_screen");
  const integer n_var{parameter.get_int("n_var")};
  real total_simulation_time{parameter.get_real("total_simulation_time")};
  const integer output_file = parameter.get_int("output_file");

  bool finished{false};
  // This should be got from a Parameter later, which may be got from a previous simulation.
  real physical_time{parameter.get_real("solution_time")};

  real dt{1e+6};
  const bool fixed_time_step{parameter.get_bool("fixed_time_step")};
  if (fixed_time_step) {
    dt = parameter.get_real("dt");
  }

  while (!finished) {
    ++step;
    if (step > total_step) {
      break;
    }

    // Start a single iteration
    // First, store the value of last step if we need to compute residual
    if (step % output_screen == 0) {
      for (auto b = 0; b < n_block; ++b) {
        store_last_step<<<bpg[b], tpb>>>(field[b].d_ptr);
      }
    }

    // For every iteration, we need to save the conservative variables of the last step.
    for (auto b = 0; b < n_block; ++b) {
//      const auto mx{mesh[b].mx + ng_1}, my{mesh[b].my + ng_1}, mz{mesh[b].mz + ng_1};
//      dim3 BPG{mx / tpb.x + 1, my / tpb.y + 1, mz / tpb.z + 1};
//      compute_cv_from_bv<mix_model, turb><<<BPG, tpb>>>(field[b].d_ptr, param);
      cudaMemcpy(field[b].h_ptr->qn.data(), field[b].h_ptr->cv.data(), field[b].h_ptr->cv.size() * sizeof(real) * n_var,
                 cudaMemcpyDeviceToDevice);
    }

    // Rk inner iteration
    for (integer rk = 0; rk < 3; ++rk) {
      for (auto b = 0; b < n_block; ++b) {
        // Set dq to 0
        cudaMemset(field[b].h_ptr->dq.data(), 0, field[b].h_ptr->dq.size() * n_var * sizeof(real));

        // Second, for each block, compute the residual dq
        // First, compute the source term, because properties such as mut are updated here.
        compute_source<mix_model, turb><<<bpg[b], tpb>>>(field[b].d_ptr, param);
        compute_inviscid_flux<mix_model, turb>(mesh[b], field[b].d_ptr, param, n_var, parameter);
        compute_viscous_flux<mix_model, turb>(mesh[b], field[b].d_ptr, param, n_var);
      }

      // For unsteady simulations, the time step should be consistent in all grid points
      if (!fixed_time_step) {
        // compute the local time step
        for (auto b = 0; b < n_block; ++b)
          local_time_step<mix_model, turb><<<bpg[b], tpb>>>(field[b].d_ptr, param);
        // After all processes and all blocks computing dt_local, we compute the global time step.
        dt = global_time_step(mesh, parameter, field);
      }

      for (auto b = 0; b < n_block; ++b) {
        // Explicit temporal schemes should not use any implicit treatment.

        // update basic variables
        update_cv_and_bv_rk<mix_model, turb><<<bpg[b], tpb>>>(field[b].d_ptr, param, dt, rk);

        // limit unphysical values computed by the program
        //limit_flow<mix_model, turb><<<bpg[b], tpb>>>(field[b].d_ptr, param, b);

        // Apply boundary conditions
        // Attention: "driver" is a template class, when a template class calls a member function of another template,
        // the compiler will not treat the called function as a template function,
        // so we need to explicitly specify the "template" keyword here.
        // If we call this function in the "driver" member function, we can omit the "template" keyword, as shown in Driver.cu, line 88.
        driver.bound_cond.template apply_boundary_conditions<mix_model, turb, true>(mesh[b], field[b], param);
      }
      // Third, transfer data between and within processes
      data_communication<mix_model, turb, true>(mesh, field, parameter, step, param);

      if (mesh.dimension == 2) {
        for (auto b = 0; b < n_block; ++b) {
          const auto mx{mesh[b].mx}, my{mesh[b].my};
          dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, 1};
          eliminate_k_gradient<true><<<BPG, tpb>>>(field[b].d_ptr, param);
        }
      }

      // update physical properties such as Mach number, transport coefficients et, al.
      for (auto b = 0; b < n_block; ++b) {
        integer mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
        dim3 BPG{(mx + 1) / tpb.x + 1, (my + 1) / tpb.y + 1, (mz + 1) / tpb.z + 1};
        update_physical_properties<mix_model><<<BPG, tpb>>>(field[b].d_ptr, param);
      }
    }

    // Finally, test if the simulation reaches convergence state
    physical_time += dt;
    if (step % output_screen == 0 || step == 1) {
      real err_max = compute_residual(driver, step);
      if (driver.myid == 0) {
        unsteady_screen_output(step, err_max, driver.time, driver.res, dt, physical_time);
      }
    }
    cudaDeviceSynchronize();
    if (physical_time > total_simulation_time) {
      finished = true;
    }
    if (step % output_file == 0 || finished) {
      ioManager.print_field(step, parameter, physical_time);
      timeSeriesIOManager.print_field(step, parameter, physical_time);
      post_process(driver);
    }
  }
  delete[] bpg;
}

template<MixtureModel mix_model, class turb_method>
__global__ void update_cv_and_bv_rk(cfd::DZone *zone, DParameter *param, real dt, integer rk) {
  const integer extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  auto &cv = zone->cv;
  auto &qn = zone->qn;

  real dt_div_jac = dt / zone->jac(i, j, k);
  for (integer l = 0; l < param->n_var; ++l) {
    cv(i, j, k, l) = SSPRK3::a[rk] * qn(i, j, k, l) + SSPRK3::b[rk] * cv(i, j, k, l) +
                     SSPRK3::c[rk] * dt_div_jac * zone->dq(i, j, k, l);
  }
  if (extent[2] == 1) {
    cv(i, j, k, 3) = 0;
  }

  auto &bv = zone->bv;
  auto &velocity = zone->vel(i, j, k);

  bv(i, j, k, 0) = cv(i, j, k, 0);
  const real density_inv = 1.0 / cv(i, j, k, 0);
  bv(i, j, k, 1) = cv(i, j, k, 1) * density_inv;
  bv(i, j, k, 2) = cv(i, j, k, 2) * density_inv;
  bv(i, j, k, 3) = cv(i, j, k, 3) * density_inv;
  velocity = bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3); //V^2

  auto &sv = zone->sv;
  if constexpr (mix_model != MixtureModel::FL) {
    // For multiple species or RANS methods, there will be scalars to be computed
    for (integer l = 0; l < param->n_scalar; ++l) {
      sv(i, j, k, l) = cv(i, j, k, 5 + l) * density_inv;
    }
  } else {
    // Flamelet model
    for (integer l = 0; l < param->n_scalar_transported; ++l) {
      sv(i, j, k, l + param->n_spec) = cv(i, j, k, 5 + l) * density_inv;
    }
    real yk_ave[MAX_SPEC_NUMBER];
    memset(yk_ave, 0, sizeof(real) * param->n_spec);
    compute_massFraction_from_MixtureFraction(zone, i, j, k, param, yk_ave);
    for (integer l = 0; l < param->n_spec; ++l) {
      sv(i, j, k, l) = yk_ave[l];
    }
  }
  if constexpr (mix_model != MixtureModel::Air) {
    compute_temperature_and_pressure(i, j, k, param, zone, cv(i, j, k, 4));
  } else {
    // Air
    bv(i, j, k, 4) = (gamma_air - 1) * (cv(i, j, k, 4) - 0.5 * bv(i, j, k, 0) * velocity);
    bv(i, j, k, 5) = bv(i, j, k, 4) * mw_air * density_inv / R_u;
  }
  velocity = std::sqrt(velocity);
}
}
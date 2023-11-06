#pragma once

#include "Driver.cuh"
#include "TimeAdvanceFunc.cuh"
#include "SourceTerm.cuh"

namespace cfd {
template<MixtureModel mix_model, class turb>
void first_order_euler_bv(Driver<mix_model, turb> &driver) {
  auto &parameter{driver.parameter};
  auto &mesh{driver.mesh};
  std::vector<cfd::Field> &field{driver.field};
  DParameter *param{driver.param};

  if (driver.myid == 0) {
    printf("Unsteady flow simulation with 1st order Euler scheme for time advancing.\n");
  }

  integer step{parameter.get_int("step")};
  integer total_step{parameter.get_int("total_step") + step};
  real total_simulation_time{parameter.get_real("total_simulation_time")};
  const integer n_block{mesh.n_block};
  const integer n_var{parameter.get_int("n_var")};
  const integer ngg{mesh[0].ngg};
  const integer ng_1 = 2 * ngg - 1;
  const integer output_screen = parameter.get_int("output_screen");
  const integer output_file = parameter.get_int("output_file");
  const bool fixed_time_step{parameter.get_bool("fixed_time_step")};

  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  dim3 *bpg = new dim3[n_block];
  for (integer b = 0; b < n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    bpg[b] = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
  }

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

  IOManager<mix_model, turb> ioManager(driver.myid, mesh, field, parameter, driver.spec, 0);
  TimeSeriesIOManager<mix_model, turb> timeSeriesIOManager(driver.myid, mesh, field, parameter, driver.spec, 0);

  bool finished{false};
  // This should be got from a Parameter later, which may be got from a previous simulation.
  real physical_time{0};
  while (!finished) {
    ++step;
    if (step > total_step) {
      break;
    }

    if constexpr (mix_model == MixtureModel::FL) {
      update_n_fl_step<<<1, 1>>>(param);
    }

    // Start a single iteration
    // First, store the value of last step
    if (step % output_screen == 0) {
      for (auto b = 0; b < n_block; ++b) {
        store_last_step<<<bpg[b], tpb>>>(field[b].d_ptr);
      }
    }
    real dt{1e+6};
    if (fixed_time_step) {
      dt = parameter.get_real("dt");
    }

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
      update_bv<mix_model, turb><<<bpg[b], tpb>>>(field[b].d_ptr, param, dt);

      // limit unphysical values computed by the program
      //limit_flow<mix_model, turb><<<bpg[b], tpb>>>(field[b].d_ptr, param, b);

      // Apply boundary conditions
      // Attention: "driver" is a template class, when a template class calls a member function of another template,
      // the compiler will not treat the called function as a template function,
      // so we need to explicitly specify the "template" keyword here.
      // If we call this function in the "driver" member function, we can omit the "template" keyword, as shown in Driver.cu, line 88.
      driver.bound_cond.template apply_boundary_conditions<mix_model, turb>(mesh[b], field[b], param);
    }
    // Third, transfer data between and within processes
    data_communication<mix_model, turb>(mesh, field, parameter, step, param);

    if (mesh.dimension == 2) {
      for (auto b = 0; b < n_block; ++b) {
        const auto mx{mesh[b].mx}, my{mesh[b].my};
        dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, 1};
        eliminate_k_gradient<<<BPG, tpb>>>(field[b].d_ptr, param);
      }
    }

    // update physical properties such as Mach number, transport coefficients et, al.
    for (auto b = 0; b < n_block; ++b) {
      integer mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
      dim3 BPG{(mx + 1) / tpb.x + 1, (my + 1) / tpb.y + 1, (mz + 1) / tpb.z + 1};
      update_physical_properties<mix_model><<<BPG, tpb>>>(field[b].d_ptr, param);
    }

    // Finally, test if the simulation reaches convergence state
    physical_time += dt;
    if (step % output_screen == 0 || step == 1) {
      real err_max = compute_residual(driver, step);
//      converged = err_max < parameter.get_real("convergence_criteria");
      if (driver.myid == 0) {
        unsteady_screen_output(step, err_max, driver.time, driver.res, dt, physical_time);
      }
    }
    cudaDeviceSynchronize();
    if (physical_time > total_simulation_time) {
      finished = true;
    }
    if (step % output_file == 0 || finished) {
      if constexpr (mix_model == MixtureModel::FL) {
        integer n_fl_step{0};
        cudaMemcpy(&n_fl_step, &(param->n_fl_step), sizeof(integer), cudaMemcpyDeviceToHost);
        parameter.update_parameter("n_fl_step", n_fl_step);
      }
      ioManager.print_field(step, parameter);
      timeSeriesIOManager.print_field(step, parameter, physical_time);
      post_process(driver);
    }
  }
  delete[] bpg;
}

template<MixtureModel mix_model, class turb>
void first_order_euler_cv(Driver<mix_model, turb> &driver) {
  // This specialization is used for methods that should reconstruct cv, such as WENO.
  // The method is not implemented yet. But the interface is left here.
  auto &parameter{driver.parameter};
  auto &mesh{driver.mesh};
  std::vector<cfd::Field> &field{driver.field};
  DParameter *param{driver.param};

  if (driver.myid == 0) {
    printf("Unsteady flow simulation with 1st order Euler scheme for time advancing.\n");
  }

  integer step{parameter.get_int("step")};
  integer total_step{parameter.get_int("total_step") + step};
  real total_simulation_time{parameter.get_real("total_simulation_time")};
  const integer n_block{mesh.n_block};
  const integer n_var{parameter.get_int("n_var")};
  const integer ngg{mesh[0].ngg};
  const integer ng_1 = 2 * ngg - 1;
  const integer output_screen = parameter.get_int("output_screen");
  const integer output_file = parameter.get_int("output_file");
  const bool fixed_time_step{parameter.get_bool("fixed_time_step")};

  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  dim3 *bpg = new dim3[n_block];
  for (integer b = 0; b < n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    bpg[b] = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
  }

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

  bool finished{false};
  while (!finished) {
    ++step;
    if (step > total_step) {
      break;
    }

    if constexpr (mix_model == MixtureModel::FL) {
      update_n_fl_step<<<1, 1>>>(param);
    }

    // Start a single iteration
    // First, store the value of last step
    if (step % output_screen == 0) {
      for (auto b = 0; b < n_block; ++b) {
        store_last_step<<<bpg[b], tpb>>>(field[b].d_ptr);
      }
    }

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
    real dt{1e+6};
    if (fixed_time_step) {
      dt = parameter.get_real("dt");
    } else {
      // compute the local time step
      for (auto b = 0; b < n_block; ++b)
        local_time_step<mix_model, turb><<<bpg[b], tpb>>>(field[b].d_ptr, param);
      // After all processes and all blocks computing dt_local, we compute the global time step.
      dt = global_time_step(mesh, parameter, field);
    }

    for (auto b = 0; b < n_block; ++b) {
      // Explicit temporal schemes should not use any implicit treatment.

      // update conservative and basic variables
      update_bv<mix_model, turb><<<bpg[b], tpb>>>(field[b].d_ptr, param, dt);

      // limit unphysical values computed by the program
      //limit_flow<mix_model, turb><<<bpg[b], tpb>>>(field[b].d_ptr, param, b);

      // Apply boundary conditions
      // Attention: "driver" is a template class, when a template class calls a member function of another template,
      // the compiler will not treat the called function as a template function,
      // so we need to explicitly specify the "template" keyword here.
      // If we call this function in the "driver" member function, we can omit the "template" keyword, as shown in Driver.cu, line 88.
      driver.bound_cond.template apply_boundary_conditions<mix_model, turb>(mesh[b], field[b], param);
    }
    // Third, transfer data between and within processes
    data_communication<mix_model, turb>(mesh, field, parameter, step, param);

    if (mesh.dimension == 2) {
      for (auto b = 0; b < n_block; ++b) {
        const auto mx{mesh[b].mx}, my{mesh[b].my};
        dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, 1};
        eliminate_k_gradient<<<BPG, tpb>>>(field[b].d_ptr, param);
      }
    }

    // update physical properties such as Mach number, transport coefficients et, al.
    for (auto b = 0; b < n_block; ++b) {
      integer mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
      dim3 BPG{(mx + 1) / tpb.x + 1, (my + 1) / tpb.y + 1, (mz + 1) / tpb.z + 1};
      update_physical_properties<mix_model><<<BPG, tpb>>>(field[b].d_ptr, param);
    }

    // Finally, test if the simulation reaches convergence state
    if (step % output_screen == 0 || step == 1) {
      real err_max = compute_residual(driver, step);
//      converged = err_max < parameter.get_real("convergence_criteria");
      if (driver.myid == 0) {
        steady_screen_output(step, err_max, driver.time, driver.res);
      }
    }
    cudaDeviceSynchronize();
    if (step % output_file == 0 || finished) {
      if constexpr (mix_model == MixtureModel::FL) {
        integer n_fl_step{0};
        cudaMemcpy(&n_fl_step, &(param->n_fl_step), sizeof(integer), cudaMemcpyDeviceToHost);
        parameter.update_parameter("n_fl_step", n_fl_step);
      }
//      ioManager.print_field(step, parameter);
      post_process(driver);
    }
  }
  delete[] bpg;
}

template<MixtureModel mix_model, class turb, class ReconstructBVorCV>
void first_order_euler(Driver<mix_model, turb> &driver) {
  if constexpr (std::is_same_v<ReconstructBVorCV, reconstruct_bv>) {
    first_order_euler_bv(driver);
  } else {
    first_order_euler_cv(driver);
  }
}

}
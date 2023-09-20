#pragma once

#include "Driver.cuh"
#include "IOManager.h"
#include "TimeAdvanceFunc.cuh"
#include "SourceTerm.cuh"
#include "SchemeSelector.cuh"
#include "ImplicitTreatmentHPP.cuh"
#include "FieldOperation.cuh"
#include "DataCommunication.cuh"
#include "Residual.cuh"
#include "PostProcess.h"

namespace cfd {
template<MixtureModel mix_model, TurbMethod turb_method>
void steady_simulation(Driver<mix_model, turb_method> &driver) {
  auto &parameter{driver.parameter};
  auto &mesh{driver.mesh};
  std::vector<cfd::Field> &field{driver.field};
  DParameter *param{driver.param};

  if (driver.myid == 0) {
    printf("Steady flow simulation.\n");
  }

  bool converged{false};
  integer step{parameter.get_int("step")};
  integer total_step{parameter.get_int("total_step") + step};
  const integer n_block{mesh.n_block};
  const integer n_var{parameter.get_int("n_var")};
  const integer ngg{mesh[0].ngg};
  const integer ng_1 = 2 * ngg - 1;
  const integer output_screen = parameter.get_int("output_screen");
  const integer output_file = parameter.get_int("output_file");

  IOManager<mix_model, turb_method> ioManager(driver.myid, mesh, field, parameter, driver.spec, 0);

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
    store_last_step<<<bpg[b], tpb>>>(field[b].d_ptr);
  }

  while (!converged) {
    ++step;
    /*[[unlikely]]*/if (step > total_step) {
      break;
    }

    if constexpr (mix_model==MixtureModel::FL){
      update_n_fl_step<<<1,1>>>(param);
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
      compute_source<mix_model, turb_method><<<bpg[b], tpb>>>(field[b].d_ptr, param);
      compute_inviscid_flux<mix_model, turb_method>(mesh[b], field[b].d_ptr, param, n_var, parameter);
      compute_viscous_flux<mix_model, turb_method>(mesh[b], field[b].d_ptr, param, n_var);

      // compute the local time step
      local_time_step<mix_model, turb_method><<<bpg[b], tpb>>>(field[b].d_ptr, param);
      // implicit treatment if needed
      implicit_treatment<mix_model, turb_method>(mesh[b], param, field[b].d_ptr, parameter, field[b].h_ptr);

      // update conservative and basic variables
      update_cv_and_bv<mix_model, turb_method><<<bpg[b], tpb>>>(field[b].d_ptr, param);

      // limit unphysical values computed by the program
      limit_flow<mix_model, turb_method><<<bpg[b], tpb>>>(field[b].d_ptr, param, b);

      // apply boundary conditions
      // Attention: "driver" is a template class, when a template class calls a member function of another template,
      // the compiler will not treat the called function as a template function,
      // so we need to explicitly specify the "template" keyword here.
      // If we call this function in the "driver" member function, we can omit the "template" keyword, as shown in Driver.cu, line 88.
      driver.bound_cond.template apply_boundary_conditions<mix_model, turb_method>(mesh[b], field[b], param);
    }
    // Third, transfer data between and within processes
    data_communication<mix_model, turb_method>(mesh, field, parameter, step, param);

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
      update_physical_properties<mix_model, turb_method><<<BPG, tpb>>>(field[b].d_ptr, param);
    }

    // Finally, test if the simulation reaches convergence state
    if (step % output_screen == 0 || step == 1) {
      real err_max = compute_residual(driver, step);
      converged = err_max < parameter.get_real("convergence_criteria");
      if (driver.myid == 0) {
        steady_screen_output(step, err_max, driver.time, driver.res);
      }
    }
    cudaDeviceSynchronize();
    if (step % output_file == 0 || converged) {
      if constexpr (mix_model==MixtureModel::FL){
        integer n_fl_step{0};
        cudaMemcpy(&n_fl_step, &(param->n_fl_step), sizeof(integer), cudaMemcpyDeviceToHost);
        parameter.update_parameter("n_fl_step", n_fl_step);
      }
      ioManager.print_field(step, parameter);
      post_process(driver);
    }
  }
  delete[] bpg;
}
}
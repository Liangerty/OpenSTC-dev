#include "Driver.cuh"
#include "Initialize.cuh"
#include "DataCommunication.cuh"
#include "TimeAdvanceFunc.cuh"
#include "SourceTerm.cuh"
#include "SchemeSelector.cuh"
#include "ImplicitTreatmentHPP.cuh"
#include "Parallel.h"
#include "PostProcess.h"
#include "MPIIO.hpp"
#include "SteadySim.cuh"
#include "WallDistance.cuh"

namespace cfd {

template<MixtureModel mix_model, TurbMethod turb_method>
Driver<mix_model, turb_method>::Driver(Parameter &parameter, Mesh &mesh_):
    myid(parameter.get_int("myid")), time(), mesh(mesh_), parameter(parameter),
    spec(parameter), reac(parameter, spec) {
  // Allocate the memory for every block
  for (integer blk = 0; blk < mesh.n_block; ++blk) {
    field.emplace_back(parameter, mesh[blk]);
  }

  initialize_basic_variables<mix_model, turb_method>(parameter, mesh, field, spec);

  if (parameter.get_int("initial") == 1) {
    // If continue from previous results, then we need the residual scales
    // If the file does not exist, then we have a trouble
    std::ifstream res_scale_in("output/message/residual_scale.txt");
    res_scale_in >> res_scale[0] >> res_scale[1] >> res_scale[2] >> res_scale[3];
    res_scale_in.close();
  }

  for (integer blk = 0; blk < mesh.n_block; ++blk) {
    field[blk].setup_device_memory(parameter);
  }
  bound_cond.initialize_bc_on_GPU(mesh_, field, spec, parameter);
  DParameter d_param(parameter, spec, &reac);
  cudaMalloc(&param, sizeof(DParameter));
  cudaMemcpy(param, &d_param, sizeof(DParameter), cudaMemcpyHostToDevice);

  write_reference_state(parameter);
}

template<MixtureModel mix_model, TurbMethod turb_method>
void Driver<mix_model, turb_method>::initialize_computation() {
  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  const auto ng_1 = 2 * mesh[0].ngg - 1;

  // If we use k-omega SST model, we need the wall distance, thus we need to compute or read it here.
  if constexpr (turb_method == TurbMethod::RANS) {
    if (parameter.get_int("RANS_model") == 2) {
      // SST method
      acquire_wall_distance<mix_model, turb_method>(*this);
    }
  }

  if (mesh.dimension == 2) {
    for (auto b = 0; b < mesh.n_block; ++b) {
      const auto mx{mesh[b].mx}, my{mesh[b].my};
      dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, 1};
      eliminate_k_gradient <<<BPG, tpb >>>(field[b].d_ptr, param);
    }
  }

  // Second, apply boundary conditions to all boundaries, including face communication between faces
  for (integer b = 0; b < mesh.n_block; ++b) {
    bound_cond.apply_boundary_conditions<mix_model, turb_method>(mesh[b], field[b], param);
  }
  if (myid == 0) {
    printf("Boundary conditions are applied successfully for initialization\n");
  }


  // First, compute the conservative variables from basic variables
  for (auto i = 0; i < mesh.n_block; ++i) {
    integer mx{mesh[i].mx}, my{mesh[i].my}, mz{mesh[i].mz};
    dim3 bpg{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
    compute_cv_from_bv<mix_model, turb_method><<<bpg, tpb>>>(field[i].d_ptr, param);
    if constexpr (turb_method == TurbMethod::RANS) {
      // We need the wall distance here. And the mut are computed for bc
      initialize_mut<mix_model><<<bpg, tpb >>>(field[i].d_ptr, param);
    }
  }
  cudaDeviceSynchronize();
  // Third, communicate values between processes
  data_communication<mix_model, turb_method>(mesh, field, parameter, 0, param);

  if (myid == 0) {
    printf("Finish data transfer.\n");
  }
  cudaDeviceSynchronize();

  for (auto b = 0; b < mesh.n_block; ++b) {
    integer mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    dim3 bpg{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
    update_physical_properties<mix_model, turb_method><<<bpg, tpb>>>(field[b].d_ptr, param);
  }
  cudaDeviceSynchronize();
  if (myid == 0) {
    printf("The flowfield is completely initialized on GPU.\n");
  }
}

//template<MixtureModel mix_model, TurbMethod turb_method>
//void Driver<mix_model, turb_method>::simulate() {
//  const auto steady{parameter.get_bool("steady")};
//  if (steady) {
//    steady_simulation<mix_model, turb_method>(*this);
//  } else {
//    const auto temporal_tag{parameter.get_int("temporal_scheme")};
//    switch (temporal_tag) {
//      case 11: // For example, if DULUSGS, then add a function to initiate the computation instead of initialize before setting up the scheme as CPU code
//        break;
//      case 12:
//        break;
//      default:
//        printf("Not implemented");
//    }
//  }
//}

//template<MixtureModel mix_model, TurbMethod turb_method>
//real Driver<mix_model, turb_method>::compute_residual(integer step) {
//  const integer n_block{mesh.n_block};
//  for (auto &e: res) {
//    e = 0;
//  }
//
//  dim3 tpb{8, 8, 4};
//  if (mesh.dimension == 2) {
//    tpb = {16, 16, 1};
//  }
//  for (integer b = 0; b < n_block; ++b) {
//    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
//    dim3 bpg = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
//    // compute the square of the difference of the basic variables
//    compute_square_of_dbv<<<bpg, tpb>>>(field[b].d_ptr);
//  }
//
//  constexpr integer TPB{128};
//  constexpr integer n_res_var{4};
//  real res_block[n_res_var];
//  int num_sms, num_blocks_per_sm;
//  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
//  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_of_dv_squared<n_res_var>, TPB,
//                                                TPB * sizeof(real) * n_res_var);
//  for (integer b = 0; b < n_block; ++b) {
//    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
//    const integer size = mx * my * mz;
//    int n_blocks = std::min(num_blocks_per_sm * num_sms, (size + TPB - 1) / TPB);
//    reduction_of_dv_squared<n_res_var> <<<n_blocks, TPB, TPB * sizeof(real) * n_res_var >>>(
//        field[b].h_ptr->bv_last.data(), size);
//    reduction_of_dv_squared<n_res_var> <<<1, TPB, TPB * sizeof(real) * n_res_var >>>(field[b].h_ptr->bv_last.data(),
//                                                                                     n_blocks);
//    cudaMemcpy(res_block, field[b].h_ptr->bv_last.data(), n_res_var * sizeof(real), cudaMemcpyDeviceToHost);
//    for (integer l = 0; l < n_res_var; ++l) {
//      res[l] += res_block[l];
//    }
//  }
//
//  if (parameter.get_bool("parallel")) {
//    // Parallel reduction
//    static std::array<double, 4> res_temp;
//    for (int i = 0; i < 4; ++i) {
//      res_temp[i] = res[i];
//    }
//    MPI_Allreduce(res_temp.data(), res.data(), 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//  }
//  for (auto &e: res) {
//    e = std::sqrt(e / mesh.n_grid_total);
//  }
//
//  if (step == 1) {
//    for (integer i = 0; i < n_res_var; ++i) {
//      res_scale[i] = res[i];
//      if (res_scale[i] < 1e-20) {
//        res_scale[i] = 1e-20;
//      }
//    }
//    const std::filesystem::path out_dir("output/message");
//    if (!exists(out_dir)) {
//      create_directories(out_dir);
//    }
//    if (myid == 0) {
//      std::ofstream res_scale_out(out_dir.string() + "/residual_scale.txt");
//      res_scale_out << res_scale[0] << '\n' << res_scale[1] << '\n' << res_scale[2] << '\n' << res_scale[3] << '\n';
//      res_scale_out.close();
//    }
//  }
//
//  for (integer i = 0; i < 4; ++i) {
//    res[i] /= res_scale[i];
//  }
//
//  // Find the maximum error of the 4 errors
//  real err_max = res[0];
//  for (integer i = 1; i < 4; ++i) {
//    if (std::abs(res[i]) > err_max) {
//      err_max = res[i];
//    }
//  }
//
//  if (myid == 0) {
//    if (isnan(err_max)) {
//      printf("Nan occurred in step %d. Stop simulation.\n", step);
//      cfd::MpiParallel::exit();
//    }
//  }
//
//  return err_max;
//}

//template<MixtureModel mix_model, TurbMethod turb_method>
//void Driver<mix_model, turb_method>::steady_screen_output(integer step, real err_max) {
//  time.get_elapsed_time();
//  FILE *history = std::fopen("history.dat", "a");
//  fprintf(history, "%d\t%11.4e\n", step, err_max);
//  fclose(history);
//
//  printf("\n%38s    converged to: %11.4e\n", "rho", res[0]);
//  printf("  n=%8d,                       V     converged to: %11.4e   \n", step, res[1]);
//  printf("  n=%8d,                       p     converged to: %11.4e   \n", step, res[2]);
//  printf("%38s    converged to: %11.4e\n", "T ", res[3]);
//  printf("CPU time for this step is %16.8fs\n", time.step_time);
//  printf("Total elapsed CPU time is %16.8fs\n", time.elapsed_time);
////  std::cout << std::format("\n{:>38}    converged to: {:>11.4e}\n", "rho", res[0]);
////  std::cout << std::format("  n={:>8},                       V     converged to: {:>11.4e}   \n", step, res[1]);
////  std::cout << std::format("  n={:>8},                       p     converged to: {:>11.4e}   \n", step, res[2]);
////  std::cout << std::format("{:>38}    converged to: {:>11.4e}\n", "T ", res[3]);
////  std::cout << std::format("CPU time for this step is {:>16.8f}s\n", time.step_time);
////  std::cout << std::format("Total elapsed CPU time is {:>16.8f}s\n", time.elapsed_time);
//}

//template<MixtureModel mix_model, TurbMethod turb_method>
//void Driver<mix_model, turb_method>::post_process() {
//  static const std::vector<integer> processes{parameter.get_int_array("post_process")};
//
//  for (auto process: processes) {
//    switch (process) {
//      case 0: // Compute the 2D cf/qw
//        wall_friction_heatflux_2d(mesh, field, parameter);
//        break;
//      default:
//        break;
//    }
//  }
//}

__global__ void compute_wall_distance(const real *wall_point_coor, DZone *zone, integer n_point_times3) {
  const integer ngg{zone->ngg}, mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = (integer) (blockDim.x * blockIdx.x + threadIdx.x) - ngg;
  integer j = (integer) (blockDim.y * blockIdx.y + threadIdx.y) - ngg;
  integer k = (integer) (blockDim.z * blockIdx.z + threadIdx.z) - ngg;
  if (i >= mx + ngg || j >= my + ngg || k >= mz + ngg) return;

  const real x{zone->x(i, j, k)}, y{zone->y(i, j, k)}, z{zone->z(i, j, k)};
  const integer n_wall_point = n_point_times3 / 3;
  auto &wall_dist = zone->wall_distance(i, j, k);
  wall_dist = 1e+6;
  for (integer l = 0; l < n_wall_point; ++l) {
    const integer idx = 3 * l;
    real d = (x - wall_point_coor[idx]) * (x - wall_point_coor[idx]) +
             (y - wall_point_coor[idx + 1]) * (y - wall_point_coor[idx + 1]) +
             (z - wall_point_coor[idx + 2]) * (z - wall_point_coor[idx + 2]);
    if (wall_dist > d) {
      wall_dist = d;
    }
  }
  wall_dist = std::sqrt(wall_dist);
}

void write_reference_state(const Parameter &parameter) {
  if (parameter.get_int("myid") == 0) {
    std::ofstream ref_state("output/message/reference_state.txt", std::ios::trunc);
    ref_state << "Reference state\n";
    ref_state << "rho_inf = " << parameter.get_real("rho_inf") << '\n';
    ref_state << "v_inf = " << parameter.get_real("v_inf") << '\n';
    ref_state << "p_inf = " << parameter.get_real("p_inf") << '\n';
    ref_state << "T_inf = " << parameter.get_real("T_inf") << '\n';
    ref_state << "M_inf = " << parameter.get_real("M_inf") << '\n';
    ref_state << "Re_unit = " << parameter.get_real("Re_unit") << '\n';
    ref_state << "mu_inf = " << parameter.get_real("mu_inf") << '\n';
    ref_state << "acoustic_speed_inf = " << parameter.get_real("v_inf") / parameter.get_real("M_inf") << '\n';
    ref_state.close();
  }
}

// Instantiate all possible drivers
template
struct Driver<MixtureModel::Air, TurbMethod::Laminar>;
template
struct Driver<MixtureModel::Air, TurbMethod::RANS>;
template
struct Driver<MixtureModel::Mixture, TurbMethod::Laminar>;
template
struct Driver<MixtureModel::Mixture, TurbMethod::RANS>;
template
struct Driver<MixtureModel::FR, TurbMethod::Laminar>;
template
struct Driver<MixtureModel::FR, TurbMethod::RANS>;

} // cfd
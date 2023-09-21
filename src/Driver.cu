#include "Driver.cuh"
#include "Initialize.cuh"
#include "DataCommunication.cuh"
#include "TimeAdvanceFunc.cuh"
#include "MPIIO.hpp"
#include "WallDistance.cuh"

namespace cfd {

template<MixtureModel mix_model, TurbulenceMethod turb_method, class turb>
Driver<mix_model, turb_method, turb>::Driver(Parameter &parameter, Mesh &mesh_):
    myid(parameter.get_int("myid")), time(), mesh(mesh_), parameter(parameter),
    spec(parameter), reac(parameter, spec) {
  // Allocate the memory for every block
  for (integer blk = 0; blk < mesh.n_block; ++blk) {
    field.emplace_back(parameter, mesh[blk]);
  }

  initialize_basic_variables<mix_model, turb>(parameter, mesh, field, spec);

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

template<MixtureModel mix_model, TurbulenceMethod turb_method, class turb>
void Driver<mix_model, turb_method, turb>::initialize_computation() {
  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  const auto ng_1 = 2 * mesh[0].ngg - 1;

  // If we use k-omega SST model, we need the wall distance, thus we need to compute or read it here.
  if constexpr (turb_method == TurbulenceMethod::RANS) {
    if (parameter.get_int("RANS_model") == 2) {
      // SST method
      acquire_wall_distance<mix_model, turb_method, turb>(*this);
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
    bound_cond.apply_boundary_conditions<mix_model, turb>(mesh[b], field[b], param);
  }
  if (myid == 0) {
    printf("Boundary conditions are applied successfully for initialization\n");
  }


  // First, compute the conservative variables from basic variables
  for (auto i = 0; i < mesh.n_block; ++i) {
    integer mx{mesh[i].mx}, my{mesh[i].my}, mz{mesh[i].mz};
    dim3 bpg{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
    compute_velocity<mix_model, turb_method><<<bpg, tpb>>>(field[i].d_ptr, param);
    if constexpr (TurbMethod<turb>::hasMut==true){
      initialize_mut<mix_model, turb><<<bpg, tpb >>>(field[i].d_ptr, param);
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
struct Driver<MixtureModel::Air, TurbulenceMethod::Laminar, Laminar>;
template
struct Driver<MixtureModel::Air, TurbulenceMethod::RANS, SST::SST>;
template
struct Driver<MixtureModel::Mixture, TurbulenceMethod::Laminar, Laminar>;
template
struct Driver<MixtureModel::Mixture, TurbulenceMethod::RANS, SST::SST>;
template
struct Driver<MixtureModel::FR, TurbulenceMethod::Laminar, Laminar>;
template
struct Driver<MixtureModel::FR, TurbulenceMethod::RANS, SST::SST>;

} // cfd
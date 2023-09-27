#pragma once

#include "Define.h"
#include "Driver.cuh"
#include <vector>

namespace cfd{
template<MixtureModel mix_model, class turb>
void acquire_wall_distance(Driver<mix_model, turb> &driver) {
  auto& parameter{driver.parameter};
  auto &bound_cond{driver.bound_cond};
  auto&mesh{driver.mesh};
  const integer myid{parameter.get_int("myid")};
  std::vector<Field>& field{driver.field};

  integer method_for_wall_distance{parameter.get_int("wall_distance")};
  if (method_for_wall_distance == 0) {
    // We need to compute it.

    // Store all wall coordinates of this process in the vector
    std::vector<real> wall_coor;
    for (integer iw = 0; iw < bound_cond.n_wall; ++iw) {
      auto &info = bound_cond.wall_info[iw];
      const auto nb = info.n_boundary;
      for (size_t m = 0; m < nb; m++) {
        auto i_zone = info.boundary[m].x;
        auto &x = mesh[i_zone].x;
        auto &y = mesh[i_zone].y;
        auto &z = mesh[i_zone].z;
        auto &f = mesh[i_zone].boundary[info.boundary[m].y];
        for (integer k = f.range_start[2]; k <= f.range_end[2]; ++k) {
          for (integer j = f.range_start[1]; j <= f.range_end[1]; ++j) {
            for (integer i = f.range_start[0]; i <= f.range_end[0]; ++i) {
              wall_coor.push_back(x(i, j, k));
              wall_coor.push_back(y(i, j, k));
              wall_coor.push_back(z(i, j, k));
            }
          }
        }
      }
    }
    const integer n_proc{parameter.get_int("n_proc")};
    auto *n_wall_point = new integer[n_proc];
    auto n_wall_this = static_cast<integer>(wall_coor.size());
    MPI_Allgather(&n_wall_this, 1, MPI_INT, n_wall_point, 1, MPI_INT, MPI_COMM_WORLD);
    auto *disp = new integer[n_proc];
    disp[0] = 0;
    for (integer i = 1; i < n_proc; ++i) {
      disp[i] = disp[i - 1] + n_wall_point[i - 1];
    }
    integer total_wall_number{0};
    for (integer i = 0; i < n_proc; ++i) {
      total_wall_number += n_wall_point[i];
    }
    std::vector<real> wall_points(total_wall_number, 0);
    // NOTE: The MPI process here is not examined carefully, if there are mistakes or things hard to understand, examine here.
    MPI_Allgatherv(wall_coor.data(), n_wall_point[myid], MPI_DOUBLE, wall_points.data(), n_wall_point, disp, MPI_DOUBLE,
                   MPI_COMM_WORLD);
    real *wall_corr_gpu = nullptr;
    cudaMalloc(&wall_corr_gpu, total_wall_number * sizeof(real));
    cudaMemcpy(wall_corr_gpu, wall_points.data(), total_wall_number * sizeof(real), cudaMemcpyHostToDevice);
    if (myid == 0) {
      printf("Start computing wall distance.\n");
    }
    for (integer blk = 0; blk < mesh.n_block; ++blk) {
      const integer ngg{mesh[0].ngg};
      const integer mx{mesh[blk].mx + 2 * ngg}, my{mesh[blk].my + 2 * ngg}, mz{mesh[blk].mz + 2 * ngg};
      dim3 tpb{512, 1, 1};
      dim3 bpg{(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
      compute_wall_distance<<<bpg, tpb>>>(wall_corr_gpu, field[blk].d_ptr, total_wall_number);
//      cudaMemcpy(field[blk].var_without_ghost_grid.data(), field[blk].h_ptr->wall_distance.data(), field[blk].h_ptr->wall_distance.size()*sizeof(real),cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    if (myid == 0) {
      printf("Finish computing wall distance.\n");
    }

  } else {
    // We need to read it.
  }
}

}
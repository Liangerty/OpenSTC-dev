#include "TimeAdvanceFunc.cuh"
#include "Field.h"
#include <mpi.h>
#include "gxl_lib/MyAtomic.cuh"

__global__ void cfd::store_last_step(cfd::DZone *zone) {
  const integer mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = blockDim.x * blockIdx.x + threadIdx.x;
  integer j = blockDim.y * blockIdx.y + threadIdx.y;
  integer k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= mx || j >= my || k >= mz) return;

  zone->bv_last(i, j, k, 0) = zone->bv(i, j, k, 0);
  zone->bv_last(i, j, k, 1) = zone->vel(i, j, k);
  zone->bv_last(i, j, k, 2) = zone->bv(i, j, k, 4);
  zone->bv_last(i, j, k, 3) = zone->bv(i, j, k, 5);
}

__global__ void cfd::compute_square_of_dbv(cfd::DZone *zone) {
  const integer mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = blockDim.x * blockIdx.x + threadIdx.x;
  integer j = blockDim.y * blockIdx.y + threadIdx.y;
  integer k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= mx || j >= my || k >= mz) return;

  auto &bv = zone->bv;
  auto &bv_last = zone->bv_last;

  bv_last(i, j, k, 0) = (bv(i, j, k, 0) - bv_last(i, j, k, 0)) * (bv(i, j, k, 0) - bv_last(i, j, k, 0));
  bv_last(i, j, k, 1) = (zone->vel(i, j, k) - bv_last(i, j, k, 1)) * (zone->vel(i, j, k) - bv_last(i, j, k, 1));
  bv_last(i, j, k, 2) = (bv(i, j, k, 4) - bv_last(i, j, k, 2)) * (bv(i, j, k, 4) - bv_last(i, j, k, 2));
  bv_last(i, j, k, 3) = (bv(i, j, k, 5) - bv_last(i, j, k, 3)) * (bv(i, j, k, 5) - bv_last(i, j, k, 3));
}

__global__ void
cfd::repair_turbulent_variables(cfd::DZone *zone, cfd::DParameter *param, integer blk_id, integer count_start) {
  const auto mx{zone->mx}, my{zone->my}, mz{zone->mz};
  auto &unphysical = zone->unphysical;
  auto &sv = zone->sv;
  const integer n_spec{param->n_spec};
  for (int k = 0; k < mz; ++k) {
    for (int j = 0; j < my; ++j) {
      for (int i = 0; i < mx; ++i) {
        if (unphysical(i, j, k)) {
          real updated_var[2];
          memset(updated_var, 0, 2 * sizeof(real));
          integer kn{0};
          // Compute the sum of all "good" points surrounding the "bad" point
          for (integer ka = -1; ka < 2; ++ka) {
            const integer k1{k + ka};
            if (k1 < 0 || k1 >= mz) continue;
            for (integer ja = -1; ja < 2; ++ja) {
              const integer j1{j + ja};
              if (j1 < 0 || j1 >= my) continue;
              for (integer ia = -1; ia < 2; ++ia) {
                const integer i1{i + ia};
                if (i1 < 0 || i1 >= mx)continue;

                if (isnan(sv(i1, j1, k1, n_spec)) || isnan(sv(i1, j1, k1, 1 + n_spec)) || sv(i1, j1, k1, n_spec) < 0 ||
                    sv(i1, j1, k1, n_spec + 1) < 0) {
                  continue;
                }

                updated_var[0] += sv(i1, j1, k1, n_spec);
                updated_var[1] += sv(i1, j1, k1, 1 + n_spec);

                ++kn;
              }
            }
          }

          // Compute the average of the surrounding points
          if (kn > 0) {
            const real kn_inv{1.0 / kn};
            updated_var[0] *= kn_inv;
            updated_var[1] *= kn_inv;
          } else {
            // The surrounding points are all "bad"
            updated_var[0] = sv(i, j, k, n_spec) < 0 ? param->limit_flow.sv_inf[n_spec] : sv(i, j, k, n_spec);
            updated_var[1] =
                sv(i, j, k, n_spec + 1) < 0 ? param->limit_flow.sv_inf[n_spec + 1] : sv(i, j, k, n_spec + 1);
          }

          // Assign averaged values for the bad point
          sv(i, j, k, n_spec) = updated_var[0];
          sv(i, j, k, n_spec + 1) = updated_var[1];

          unphysical(i, j, k) = 0;
        }
      }
    }
  }
}

__global__ void
cfd::repair_mixtureFraction_variables(cfd::DZone *zone, cfd::DParameter *param, integer blk_id, integer count_start) {
  const auto mx{zone->mx}, my{zone->my}, mz{zone->mz};
  auto &unphysical = zone->unphysical;
  auto &sv = zone->sv;
  const integer i_fl{param->i_fl};
  for (int k = 0; k < mz; ++k) {
    for (int j = 0; j < my; ++j) {
      for (int i = 0; i < mx; ++i) {
        if (unphysical(i, j, k)) {
          real updated_var[2];
          memset(updated_var, 0, 2 * sizeof(real));
          integer kn{0};
          // Compute the sum of all "good" points surrounding the "bad" point
          for (integer ka = -1; ka < 2; ++ka) {
            const integer k1{k + ka};
            if (k1 < 0 || k1 >= mz) continue;
            for (integer ja = -1; ja < 2; ++ja) {
              const integer j1{j + ja};
              if (j1 < 0 || j1 >= my) continue;
              for (integer ia = -1; ia < 2; ++ia) {
                const integer i1{i + ia};
                if (i1 < 0 || i1 >= mx)continue;

                if (isnan(sv(i1, j1, k1, i_fl)) || sv(i1, j1, k1, i_fl) < 0 || sv(i1, j1, k1, i_fl) > 1
                    || isnan(sv(i1, j1, k1, 1 + i_fl)) || sv(i1, j1, k1, i_fl + 1) < 0 ||
                    sv(i1, j1, k1, i_fl + 1) > 0.25) {
                  continue;
                }

                updated_var[0] += sv(i1, j1, k1, i_fl);
                updated_var[1] += sv(i1, j1, k1, 1 + i_fl);

                ++kn;
              }
            }
          }

          // Compute the average of the surrounding points
          if (kn > 0) {
            const real kn_inv{1.0 / kn};
            updated_var[0] *= kn_inv;
            updated_var[1] *= kn_inv;
          } else {
            // The surrounding points are all "bad"
            updated_var[0] = min(1.0, max(0.0, sv(i, j, k, i_fl)));
            updated_var[1] = min(0.25, max(0.0, sv(i, j, k, i_fl + 1)));
          }

          // Assign averaged values for the bad point
          sv(i, j, k, i_fl) = updated_var[0];
          sv(i, j, k, i_fl + 1) = updated_var[1];

          unphysical(i, j, k) = 0;
        }
      }
    }
  }
}

real cfd::global_time_step(const Mesh &mesh, const Parameter &parameter, std::vector<cfd::Field> &field) {
  real dt{1e+6};

  constexpr integer TPB{128};
  real dt_block;
  int num_sms, num_blocks_per_sm;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, min_of_arr, TPB, 0);
  for (integer b = 0; b < mesh.n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    const integer size = mx * my * mz;
    int n_blocks = std::min(num_blocks_per_sm * num_sms, (size + TPB - 1) / TPB);
    min_of_arr<<<n_blocks, TPB>>>(field[b].h_ptr->dt_local.data(), size);//, TPB * sizeof(real)
    min_of_arr<<<1, TPB>>>(field[b].h_ptr->dt_local.data(), n_blocks);//, TPB * sizeof(real)
    cudaMemcpy(&dt_block, field[b].h_ptr->dt_local.data(), sizeof(real), cudaMemcpyDeviceToHost);
    dt = std::min(dt, dt_block);
  }

  if (parameter.get_bool("parallel")) {
    // Parallel reduction
    real dt_temp{dt};
    MPI_Allreduce(&dt_temp, &dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  }

  return dt;
}

__global__ void cfd::min_of_arr(real *arr, integer size) {
  integer i = blockDim.x * blockIdx.x + threadIdx.x;
  const integer t = threadIdx.x;

  if (i >= size) {
    return;
  }
  real inp{1e+6};
  for (integer idx = i; idx < size; idx += blockDim.x * gridDim.x) {
    inp = min(inp, arr[idx]);
  }
  __syncthreads();

  inp = block_reduce_min(inp, i, size);
  __syncthreads();

  if (t == 0) {
    arr[blockIdx.x] = inp;
  }
}

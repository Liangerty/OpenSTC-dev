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

real cfd::global_time_step(const Mesh &mesh, const Parameter &parameter, std::vector<cfd::Field> &field) {
  real dt{1e+6};

  constexpr integer TPB{128};
  real dt_block;
  int num_sms, num_blocks_per_sm;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, min_of_arr, TPB, 0);
//  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, min_of_arr, TPB, TPB * sizeof(real));
  for (integer b = 0; b < mesh.n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    const integer size = mx * my * mz;
    int n_blocks = std::min(num_blocks_per_sm * num_sms, (size + TPB - 1) / TPB);
    min_of_arr<<<n_blocks, TPB>>>(field[b].h_ptr->dt_local.data(), size);//, TPB * sizeof(real)
    min_of_arr<<<1, TPB>>>(field[b].h_ptr->dt_local.data(), size);//, TPB * sizeof(real)
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
//  extern __shared__ real s[];
//  s[t] = 0;
  if (i >= size) {
    return;
  }
  real inp{1e+6};
  for (integer idx = i; idx < size; idx += blockDim.x * gridDim.x) {
    inp = min(inp, arr[idx]);
  }
//  s[t] = inp;
  __syncthreads();

  inp = block_reduce_min(inp, size);
  __syncthreads();

//  for (int stride = blockDim.x / 2, lst = blockDim.x & 1; stride >= 1; lst = stride & 1, stride >>= 1) {
//    stride += lst;
//    __syncthreads();
//    if (t < stride) {
//      //when t+stride is larger than #elements, there's no meaning of comparison. So when it happens, just keep the current value for parMax[t]. This always happens when an odd number of t satisfying the condition.
//      if (t + stride < size) {
//        s[t] = min(s[t], s[t + stride]);
//      }
//    }
//    __syncthreads();
//  }

  if (t == 0) {
//    arr[blockIdx.x] = s[0];
    arr[blockIdx.x] = inp;
  }
}

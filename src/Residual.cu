//
// Created by gxl98 on 2023/9/13.
//

#include "Residual.cuh"
#include "gxl_lib/MyAtomic.cuh"

namespace cfd {
//__inline__ __device__ real warp_reduce_sum(real val, integer idx, integer size) {
//  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
//    const auto mask = __activemask();
////    const auto mask = __ballot_sync(0xffffffff, idx + offset < size);
////    if (idx + offset < size)
//    val += __shfl_down_sync(mask, val, offset);
////    val += __shfl_down_sync(mask, val, offset);
//  }
//  return val;
//}
//
//__inline__ __device__ real block_reduce_sum(real val, integer idx, integer size) {
//  static __shared__ real shared[32];
////  memset(shared, 0, 32 * sizeof(real));
//  const auto lane = threadIdx.x % warpSize;
//  const auto wid = threadIdx.x / warpSize;
////  if (blockIdx.x == 0 && lane == 0 && wid == 0) {
////    printf("Before warp reduce: %d %e\n", threadIdx.x, val);
////  }
//  val = warp_reduce_sum(val, idx, size);
////  if (blockIdx.x == 0 && lane == 0 && wid == 0) {
////    printf("After warp reduce: %d %e\n", threadIdx.x, val);
////  }
//  if (lane == 0) {
//    shared[wid] = val;
//  }
//  __syncthreads();
//  auto n_warp{ blockDim.x / warpSize };
//  if (blockIdx.x == gridDim.x - 1) {
//    n_warp = min(n_warp, (size - (gridDim.x - 1) * blockDim.x) / warpSize + 1);
//  }
//  val = (threadIdx.x < n_warp) ? shared[lane] : 0;
//  if (wid == 0) {
//    val = warp_reduce_sum(val, lane, blockDim.x / warpSize);
////    if (blockIdx.x == 0 && lane == 0)
////      printf("After block reduce: %d %e\n", threadIdx.x, val);
//  }
//  return val;
//}

__global__ void reduction_of_dv_squared(real *arr, integer size) {
  integer i = blockDim.x * blockIdx.x + threadIdx.x;
  const integer t = threadIdx.x;
//  extern __shared__ real s[];
//  memset(&s[t * N], 0, N * sizeof(real));
  real inp[4];
  memset(inp, 0, 4 * sizeof(real));
  if (i >= size) {
    return;
  }
  for (integer idx = i; idx < size; idx += blockDim.x * gridDim.x) {
    inp[0] += arr[idx];
    inp[1] += arr[idx + size];
    inp[2] += arr[idx + size * 2];
    inp[3] += arr[idx + size * 3];
//    if (size == 149) {
//      printf("Add arr[idx=%d]: %e %e %e %e\n", idx, arr[idx], arr[idx + size], arr[idx + size * 2],
//             arr[idx + size * 3]);
//    }
  }
//  if (i == 0) {
//    printf("Before block reduce: %d %e %e %e %e\n", blockIdx.x, inp[0], inp[1], inp[2], inp[3]);
//  }
  __syncthreads();

  for (real &l: inp) {
    l = block_reduce_sum(l, size);//, i
  }
//  if (i == 0) {
//    printf("After block reduce: %d %e %e %e %e\n", blockIdx.x, inp[0], inp[1], inp[2], inp[3]);
//  }
  __syncthreads();

  if (t == 0) {
    arr[blockIdx.x] = inp[0];
    arr[blockIdx.x + gridDim.x] = inp[1];
    arr[blockIdx.x + gridDim.x * 2] = inp[2];
    arr[blockIdx.x + gridDim.x * 3] = inp[3];
//    printf("After block reduce: %d %e %e %e %e\n", blockIdx.x, inp[0], inp[1], inp[2], inp[3]);
  }
}

void steady_screen_output(integer step, real err_max, gxl::Time &time, std::array<real, 4> &res) {
  time.get_elapsed_time();
  FILE *history = std::fopen("history.dat", "a");
  fprintf(history, "%d\t%11.4e\n", step, err_max);
  fclose(history);

  printf("\n%38s    converged to: %11.4e\n", "rho", res[0]);
  printf("  n=%8d,                       V     converged to: %11.4e   \n", step, res[1]);
  printf("  n=%8d,                       p     converged to: %11.4e   \n", step, res[2]);
  printf("%38s    converged to: %11.4e\n", "T ", res[3]);
  printf("CPU time for this step is %16.8fs\n", time.step_time);
  printf("Total elapsed CPU time is %16.8fs\n", time.elapsed_time);
}

void unsteady_screen_output(integer step, real err_max, gxl::Time &time, std::array<real, 4> &res, real dt,
                            real solution_time) {
  time.get_elapsed_time();
  FILE *history = std::fopen("history.dat", "a");
  fprintf(history, "%d\t%11.4e\n", step, err_max);
  fclose(history);

  printf("\n%38s    converged to: %11.4e\n", "rho", res[0]);
  printf("  n=%8d,   dt=%13.7e,   V     converged to: %11.4e   \n", step, dt, res[1]);
  printf("  n=%8d,   dt=%13.7e,   p     converged to: %11.4e   \n", step, dt, res[2]);
  printf("%38s    converged to: %11.4e\n", "T ", res[3]);
  printf("Current physical  time is %16.8es\n", solution_time);
  printf("CPU time for this step is %16.8fs\n", time.step_time);
  printf("Total elapsed CPU time is %16.8fs\n", time.elapsed_time);
}
} // cfd
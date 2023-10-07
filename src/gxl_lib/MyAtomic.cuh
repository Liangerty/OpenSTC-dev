#pragma once

namespace cfd {
__inline__ __device__ double warp_reduce_sum(double val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    const auto mask = __activemask();
    val += __shfl_down_sync(mask, val, offset);
  }
  return val;
}

__inline__ __device__ double block_reduce_sum(double val, int size) {
  static __shared__ double shared[32];
//  memset(shared, 0, 32 * sizeof(real));
  const auto lane = threadIdx.x % warpSize;
  const auto wid = threadIdx.x / warpSize;
  val = warp_reduce_sum(val);
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  auto n_warp{blockDim.x / warpSize};
  if (blockIdx.x == gridDim.x - 1) {
    n_warp = min(n_warp, (size - (gridDim.x - 1) * blockDim.x) / warpSize + 1);
  }
  val = (threadIdx.x < n_warp) ? shared[lane] : 0;
  if (wid == 0) {
    val = warp_reduce_sum(val);
  }
  return val;
}

__inline__ __device__ double warp_reduce_min(double val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    const auto mask = __activemask();
    val = min(val, __shfl_down_sync(mask, val, offset));
  }
  return val;
}

__inline__ __device__ double block_reduce_min(double val, int size) {
  static __shared__ double shared[32];
//  memset(shared, 0, 32 * sizeof(real));
  const auto lane = threadIdx.x % warpSize;
  const auto wid = threadIdx.x / warpSize;
  val = warp_reduce_min(val);
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  auto n_warp{blockDim.x / warpSize};
  if (blockIdx.x == gridDim.x - 1) {
    n_warp = min(n_warp, (size - (gridDim.x - 1) * blockDim.x) / warpSize + 1);
  }
  val = (threadIdx.x < n_warp) ? shared[lane] : 0;
  if (wid == 0) {
    val = warp_reduce_min(val);
  }
  return val;
}
}
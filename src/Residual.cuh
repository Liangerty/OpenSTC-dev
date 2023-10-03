#pragma once

#include "Driver.cuh"
#include "TimeAdvanceFunc.cuh"
#include "mpi.h"
#include <filesystem>
#include <fstream>

namespace cfd {

//template<integer N>
//__global__ void reduction_of_dv_squared(real *arr, integer size) {
//  integer i = blockDim.x * blockIdx.x + threadIdx.x;
//  const integer t = threadIdx.x;
//  extern __shared__ real s[];
//  memset(&s[t * N], 0, N * sizeof(real));
//  if (i >= size) {
//    return;
//  }
//  real inp[N];
//  memset(inp, 0, N * sizeof(real));
//  for (integer idx = i; idx < size; idx += blockDim.x * gridDim.x) {
//    inp[0] += arr[idx];
//    inp[1] += arr[idx + size];
//    inp[2] += arr[idx + size * 2];
//    inp[3] += arr[idx + size * 3];
//  }
//  for (integer l = 0; l < N; ++l) {
//    s[t * N + l] = inp[l];
//  }
//  __syncthreads();
//
//  for (int stride = blockDim.x / 2, lst = blockDim.x & 1; stride >= 1; lst = stride & 1, stride >>= 1) {
//    stride += lst;
//    __syncthreads();
//    if (t < stride) {
//      //when t+stride is larger than #elements, there's no meaning of comparison. So when it happens, just keep the current value for parMax[t]. This always happens when an odd number of t satisfying the condition.
//      if (t + stride < size) {
//#pragma unroll
//        for (integer l = 0; l < N; ++l) {
//          s[t * N + l] += s[(t + stride) * N + l];
//        }
//      }
//    }
//    __syncthreads();
//  }
//
//  if (t == 0) {
//    arr[blockIdx.x] = s[0];
//    arr[blockIdx.x + gridDim.x] = s[1];
//    arr[blockIdx.x + gridDim.x * 2] = s[2];
//    arr[blockIdx.x + gridDim.x * 3] = s[3];
//  }
//}
__global__ void reduction_of_dv_squared(real *arr, integer size);

template<MixtureModel mix_model, class turb>
real compute_residual(Driver<mix_model, turb> &driver, integer step) {
  const auto &mesh{driver.mesh};
  std::array<real, 4> &res{driver.res};

  const integer n_block{mesh.n_block};
  for (auto &e: res) {
    e = 0;
  }

  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  std::vector<Field> &field{driver.field};
  for (integer b = 0; b < n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    dim3 bpg = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
    // compute the square of the difference of the basic variables
    compute_square_of_dbv<<<bpg, tpb>>>(field[b].d_ptr);
  }

  constexpr integer TPB{128};
  constexpr integer n_res_var{4};
  real res_block[n_res_var];
  int num_sms, num_blocks_per_sm;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_of_dv_squared, TPB, 0);
//  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_of_dv_squared<n_res_var>, TPB,
//                                                TPB * sizeof(real) * n_res_var);
  for (integer b = 0; b < n_block; ++b) {
//    int b=1;
    auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    integer size = mx * my * mz;
    int n_blocks = std::min(num_blocks_per_sm * num_sms, (size + TPB - 1) / TPB);
    reduction_of_dv_squared<<<n_blocks, TPB>>>(field[b].h_ptr->bv_last.data(), size);
    reduction_of_dv_squared<<<1, TPB>>>(field[b].h_ptr->bv_last.data(), n_blocks);
//    reduction_of_dv_squared<n_res_var> <<<n_blocks, TPB>>>(
//        field[b].h_ptr->bv_last.data(), size);//, TPB * sizeof(real) * n_res_var
//    reduction_of_dv_squared<n_res_var> <<<1, TPB>>>(field[b].h_ptr->bv_last.data(),
//                                                    n_blocks);//, TPB * sizeof(real) * n_res_var
    cudaMemcpy(res_block, field[b].h_ptr->bv_last.data(), n_res_var * sizeof(real), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (integer l = 0; l < n_res_var; ++l) {
      res[l] += res_block[l];
//      printf("b=%d, res[%d]=%e\n", b, l, res_block[l]);
    }
  }

  auto &parameter{driver.parameter};
  if (parameter.get_bool("parallel")) {
    // Parallel reduction
    static std::array<double, 4> res_temp;
    for (int i = 0; i < 4; ++i) {
      res_temp[i] = res[i];
    }
    MPI_Allreduce(res_temp.data(), res.data(), 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
//  printf("total grid number:%d\n", mesh.n_grid_total);
  for (auto &e: res) {
//    printf("before, e=%e\n", e);
    e = std::sqrt(e / mesh.n_grid_total);
//    printf("after, e=%e\n", e);
  }

  std::array<real, 4> &res_scale{driver.res_scale};
  if (step == 1) {
    for (integer i = 0; i < n_res_var; ++i) {
      res_scale[i] = res[i];
      if (res_scale[i] < 1e-20) {
        res_scale[i] = 1e-20;
      }
    }
    const std::filesystem::path out_dir("output/message");
    if (!exists(out_dir)) {
      create_directories(out_dir);
    }
    if (driver.myid == 0) {
      std::ofstream res_scale_out(out_dir.string() + "/residual_scale.txt");
      res_scale_out << res_scale[0] << '\n' << res_scale[1] << '\n' << res_scale[2] << '\n' << res_scale[3] << '\n';
      res_scale_out.close();
    }
  }

  for (integer i = 0; i < 4; ++i) {
    res[i] /= res_scale[i];
  }

  // Find the maximum error of the 4 errors
  real err_max = res[0];
  for (integer i = 1; i < 4; ++i) {
    if (std::abs(res[i]) > err_max) {
      err_max = res[i];
    }
  }

  if (driver.myid == 0) {
    if (isnan(err_max)) {
      printf("Nan occurred in step %d. Stop simulation.\n", step);
      cfd::MpiParallel::exit();
    }
  }

  return err_max;
}

void steady_screen_output(integer step, real err_max, gxl::Time &time, std::array<real, 4> &res);

} // cfd

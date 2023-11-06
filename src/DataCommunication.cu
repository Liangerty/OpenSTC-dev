#include "DataCommunication.cuh"

//void cfd::data_communication(const Mesh &mesh, std::vector<Field> &field, const Parameter &parameter, integer step,
//                        DParameter *param) {
//  // -1 - inner faces
//  for (auto blk = 0; blk < mesh.n_block; ++blk) {
//    auto &inF = mesh[blk].inner_face;
//    const auto n_innFace = inF.size();
//    auto v = field[blk].d_ptr;
//    const auto ngg = mesh[blk].ngg;
//    for (auto l = 0; l < n_innFace; ++l) {
//      // reference to the current face
//      const auto &fc = mesh[blk].inner_face[l];
//      uint tpb[3], bpg[3];
//      for (size_t j = 0; j < 3; ++j) {
//        tpb[j] = fc.n_point[j] <= (2 * ngg + 1) ? 1 : 16;
//        bpg[j] = (fc.n_point[j] - 1) / tpb[j] + 1;
//      }
//      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
//
//      // variables of the neighbor block
//      auto nv = field[fc.target_block].d_ptr;
//      inner_communication<<<BPG, TPB>>>(v, nv, l, param);
//    }
//  }
//
//  // Parallel communication via MPI
//  if (parameter.get_bool("parallel")) {
//    parallel_communication(mesh, field, step, param);
//  }
//}

__global__ void cfd::setup_data_to_be_sent(cfd::DZone *zone, integer i_face, real *data, const DParameter *param) {
  const auto &f = zone->parface[i_face];
  integer n[3];
  n[0] = blockIdx.x * blockDim.x + threadIdx.x;
  n[1] = blockDim.y * blockIdx.y + threadIdx.y;
  n[2] = blockIdx.z * blockDim.z + threadIdx.z;
  if (n[0] >= f.n_point[0] || n[1] >= f.n_point[1] || n[2] >= f.n_point[2]) return;

  integer idx[3];
  for (int ijk: f.loop_order) {
    idx[ijk] = f.range_start[ijk] + n[ijk] * f.loop_dir[ijk];
  }

  const integer n_var{param->n_scalar + 6}, ngg{zone->ngg};
  integer bias = n_var * (ngg + 1) * (n[f.loop_order[1]] * f.n_point[f.loop_order[2]] + n[f.loop_order[2]]);

  const auto &bv = zone->bv;
#pragma unroll
  for (integer l = 0; l < 6; ++l) {
    data[bias + l] = bv(idx[0], idx[1], idx[2], l);
  }
  const auto &sv = zone->sv;
  for (integer l = 0; l < param->n_scalar; ++l) {
    data[bias + 6 + l] = sv(idx[0], idx[1], idx[2], l);
  }

  for (integer ig = 1; ig <= ngg; ++ig) {
    idx[f.face] -= f.direction;
    bias += n_var;
#pragma unroll
    for (integer l = 0; l < 6; ++l) {
      data[bias + l] = bv(idx[0], idx[1], idx[2], l);
    }
    for (integer l = 0; l < param->n_scalar; ++l) {
      data[bias + 6 + l] = sv(idx[0], idx[1], idx[2], l);
    }
  }
}

//__global__ void
//cfd::inner_communication(cfd::DZone *zone, cfd::DZone *tar_zone, integer i_face, cfd::DParameter *param) {
//  const auto &f = zone->innerface[i_face];
//  uint n[3];
//  n[0] = blockIdx.x * blockDim.x + threadIdx.x;
//  n[1] = blockDim.y * blockIdx.y + threadIdx.y;
//  n[2] = blockIdx.z * blockDim.z + threadIdx.z;
//  if (n[0] >= f.n_point[0] || n[1] >= f.n_point[1] || n[2] >= f.n_point[2]) return;
//
//  integer idx[3], idx_tar[3], d_idx[3];
//  for (integer i = 0; i < 3; ++i) {
//    d_idx[i] = f.loop_dir[i] * (integer) (n[i]);
//    idx[i] = f.range_start[i] + d_idx[i];
//  }
//  for (integer i = 0; i < 3; ++i) {
//    idx_tar[i] = f.target_start[i] + f.target_loop_dir[i] * d_idx[f.src_tar[i]];
//  }
//
//  // The face direction: which of i(0)/j(1)/k(2) is the coincided face.
//  const auto face_dir{f.direction > 0 ? f.range_start[f.face] : f.range_end[f.face]};
//
//  if (idx[f.face] == face_dir) {
//    // If this is the corresponding face, then average the values from both blocks
//#pragma unroll
//    for (integer l = 0; l < 6; ++l) {
//      const real ave{0.5 * (tar_zone->bv(idx_tar[0], idx_tar[1], idx_tar[2], l) + zone->bv(idx[0], idx[1], idx[2], l))};
//      zone->bv(idx[0], idx[1], idx[2], l) = ave;
//      tar_zone->bv(idx_tar[0], idx_tar[1], idx_tar[2], l) = ave;
//    }
//    for (integer l = 0; l < param->n_scalar; ++l) {
//      const real ave{0.5 * (tar_zone->sv(idx_tar[0], idx_tar[1], idx_tar[2], l) + zone->sv(idx[0], idx[1], idx[2], l))};
//      zone->sv(idx[0], idx[1], idx[2], l) = ave;
//      tar_zone->sv(idx_tar[0], idx_tar[1], idx_tar[2], l) = ave;
//    }
//    const real vel{std::sqrt(zone->bv(idx[0], idx[1], idx[2], 1) * zone->bv(idx[0], idx[1], idx[2], 1) +
//                             zone->bv(idx[0], idx[1], idx[2], 2) * zone->bv(idx[0], idx[1], idx[2], 2) +
//                             zone->bv(idx[0], idx[1], idx[2], 3) * zone->bv(idx[0], idx[1], idx[2], 3))};
//    zone->vel(idx[0], idx[1], idx[2]) = vel;
//    tar_zone->vel(idx_tar[0], idx_tar[1], idx_tar[2]) = vel;
//  } else {
//    // Else, get the inner value for this block's ghost grid
//#pragma unroll
//    for (int l = 0; l < 6; ++l) {
//      zone->bv(idx[0], idx[1], idx[2], l) = tar_zone->bv(idx_tar[0], idx_tar[1], idx_tar[2], l);
//    }
//    for (int l = 0; l < param->n_scalar; ++l) {
//      zone->sv(idx[0], idx[1], idx[2], l) = tar_zone->sv(idx_tar[0], idx_tar[1], idx_tar[2], l);
//    }
//  }
//}

//template<MixtureModel mix_model, class turb, bool with_cv>
//__global__ void cfd::assign_data_received(cfd::DZone *zone, integer i_face, const real *data, cfd::DParameter *param) {
//  const auto &f = zone->parface[i_face];
//  integer n[3];
//  n[0] = blockIdx.x * blockDim.x + threadIdx.x;
//  n[1] = blockDim.y * blockIdx.y + threadIdx.y;
//  n[2] = blockIdx.z * blockDim.z + threadIdx.z;
//  if (n[0] >= f.n_point[0] || n[1] >= f.n_point[1] || n[2] >= f.n_point[2]) return;
//
//  integer idx[3];
//  for (int ijk: f.loop_order) {
//    idx[ijk] = f.range_start[ijk] + n[ijk] * f.loop_dir[ijk];
//  }
//
//  const integer n_var{param->n_scalar + 6}, ngg{zone->ngg};
//  integer bias = n_var * (ngg + 1) * (n[f.loop_order[1]] * f.n_point[f.loop_order[2]] + n[f.loop_order[2]]);
//
//  auto &bv = zone->bv;
//#pragma unroll
//  for (integer l = 0; l < 6; ++l) {
//    bv(idx[0], idx[1], idx[2], l) = 0.5 * (bv(idx[0], idx[1], idx[2], l) + data[bias + l]);
//  }
//  auto &sv = zone->sv;
//  for (integer l = 0; l < param->n_scalar; ++l) {
//    sv(idx[0], idx[1], idx[2], l) = 0.5 * (sv(idx[0], idx[1], idx[2], l) + data[bias + 6 + l]);
//  }
//  zone->vel(idx[0], idx[1], idx[2]) = std::sqrt(bv(idx[0], idx[1], idx[2], 1) * bv(idx[0], idx[1], idx[2], 1) +
//                                                bv(idx[0], idx[1], idx[2], 2) * bv(idx[0], idx[1], idx[2], 2) +
//                                                bv(idx[0], idx[1], idx[2], 3) * bv(idx[0], idx[1], idx[2], 3));
//
//  for (integer ig = 1; ig <= ngg; ++ig) {
//    idx[f.face] += f.direction;
//    bias += n_var;
//#pragma unroll
//    for (integer l = 0; l < 6; ++l) {
//      bv(idx[0], idx[1], idx[2], l) = data[bias + l];
//    }
//    for (integer l = 0; l < param->n_scalar; ++l) {
//      sv(idx[0], idx[1], idx[2], l) = data[bias + 6 + l];
//    }
//  }
//
//}

//void cfd::parallel_communication(const cfd::Mesh &mesh, std::vector<cfd::Field> &field, integer step,
//                                 cfd::DParameter *param) {
//  const int n_block{mesh.n_block};
//  const integer n_trans{param->n_scalar + 6}; // All primitive variables are to be transferred
//  const int ngg{mesh[0].ngg};
//  //Add up to the total face number
//  size_t total_face = 0;
//  for (int m = 0; m < n_block; ++m) {
//    total_face += mesh[m].parallel_face.size();
//  }
//
//  //A 2-D array which is the cache used when using MPI to send/recv messages. The first dimension is the face index
//  //while the second dimension is the coordinate of that face, 3 consecutive number represents one position.
//  static const auto temp_s = new real *[total_face], temp_r = new real *[total_face];
//  static const auto length = new integer[total_face];
//
//  //Added with iterate through faces and will equal to the total face number when the loop ends
//  int fc_num = 0;
//  //Compute the array size of different faces and allocate them. Different for different faces.
//  if (step == 0) {
//    for (int blk = 0; blk < n_block; ++blk) {
//      auto &B = mesh[blk];
//      const int fc = static_cast<int>(B.parallel_face.size());
//      for (int f = 0; f < fc; ++f) {
//        const auto &face = B.parallel_face[f];
//        //The length of the array is ${number of grid points of the face}*(ngg+1)*n_trans
//        //ngg+1 is the number of layers to communicate, n_trans for n_trans variables
//        const int len = n_trans * (ngg + 1) * (std::abs(face.range_start[0] - face.range_end[0]) + 1)
//                        * (std::abs(face.range_end[1] - face.range_start[1]) + 1)
//                        * (std::abs(face.range_end[2] - face.range_start[2]) + 1);
//        length[fc_num] = len;
//        cudaMalloc(&(temp_s[fc_num]), len * sizeof(real));
//        cudaMalloc(&(temp_r[fc_num]), len * sizeof(real));
//        ++fc_num;
//      }
//    }
//  }
//
//  // Create array for MPI_ISEND/IRecv
//  // MPI_REQUEST is an array representing whether the face sends/recvs successfully
//  const auto s_request = new MPI_Request[total_face], r_request = new MPI_Request[total_face];
//  const auto s_status = new MPI_Status[total_face], r_status = new MPI_Status[total_face];
//  fc_num = 0;
//
//  for (int m = 0; m < n_block; ++m) {
//    auto &B = mesh[m];
//    const int f_num = static_cast<int>(B.parallel_face.size());
//    for (int f = 0; f < f_num; ++f) {
//      //Iterate through the faces
//      const auto &fc = B.parallel_face[f];
//
//      uint tpb[3], bpg[3];
//      for (size_t j = 0; j < 3; ++j) {
//        tpb[j] = fc.n_point[j] <= (2 * ngg + 1) ? 1 : 16;
//        bpg[j] = (fc.n_point[j] - 1) / tpb[j] + 1;
//      }
//      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
//      setup_data_to_be_sent<<<BPG, TPB>>>(field[m].d_ptr, f, &temp_s[fc_num][0], param);
//      cudaDeviceSynchronize();
//      //Send and receive. Take care of the first address!
//      // The buffer is on GPU, thus we require a CUDA-aware MPI, such as OpenMPI.
//      MPI_Isend(&temp_s[fc_num][0], length[fc_num], MPI_DOUBLE, fc.target_process, fc.flag_send, MPI_COMM_WORLD,
//                &s_request[fc_num]);
//      MPI_Irecv(&temp_r[fc_num][0], length[fc_num], MPI_DOUBLE, fc.target_process, fc.flag_receive, MPI_COMM_WORLD,
//                &r_request[fc_num]);
//      ++fc_num;
//    }
//  }
//
//  //Wait for all faces finishing communication
//  MPI_Waitall(static_cast<int>(total_face), s_request, s_status);
//  MPI_Waitall(static_cast<int>(total_face), r_request, r_status);
//  MPI_Barrier(MPI_COMM_WORLD);
//
//  //Assign the correct value got by MPI receive
//  fc_num = 0;
//  for (int blk = 0; blk < n_block; ++blk) {
//    auto &B = mesh[blk];
//    const size_t f_num = B.parallel_face.size();
//    for (size_t f = 0; f < f_num; ++f) {
//      const auto &fc = B.parallel_face[f];
//      uint tpb[3], bpg[3];
//#pragma unroll
//      for (size_t j = 0; j < 3; ++j) {
//        tpb[j] = fc.n_point[j] <= (2 * ngg + 1) ? 1 : 16;
//        bpg[j] = (fc.n_point[j] - 1) / tpb[j] + 1;
//      }
//      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
//      assign_data_received<<<BPG, TPB>>>(field[blk].d_ptr, f, &temp_r[fc_num][0], param);
//      cudaDeviceSynchronize();
//      fc_num++;
//    }
//  }
//
//  //Free dynamic allocated memory
//  delete[]s_status;
//  delete[]r_status;
//  delete[]s_request;
//  delete[]r_request;
//
//}


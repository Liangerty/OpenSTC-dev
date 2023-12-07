#include "Monitor.cuh"
#include "gxl_lib/MyString.h"
#include <filesystem>
#include "Parallel.h"
#include "Field.h"

namespace cfd {
Monitor::Monitor(const Parameter &parameter, const Species &species) : if_monitor{parameter.get_int("if_monitor")},
                                                                       output_file{parameter.get_int("output_file")},
                                                                       n_block{parameter.get_int("n_block")},
                                                                       n_point(n_block, 0) {
  if (!if_monitor) {
    return;
  }

  h_ptr = new DeviceMonitorData;

  // Set up the labels to monitor
  auto var_name_found{setup_labels_to_monitor(parameter, species)};
  const auto myid{parameter.get_int("myid")};
  if (myid == 0) {
    printf("The following variables will be monitored:\n");
    for (const auto &name: var_name_found) {
      printf("%s\t", name.c_str());
    }
    printf("\n");
  }

  // Read the points to be monitored
  auto monitor_file_name{parameter.get_string("monitor_file")};
  std::filesystem::path monitor_path{monitor_file_name};
  if (!std::filesystem::exists(monitor_path)) {
    if (myid == 0) {
      printf("The monitor file %s does not exist.\n", monitor_file_name.c_str());
    }
    MpiParallel::exit();
  }
  std::ifstream monitor_file{monitor_file_name};
  std::string line;
  gxl::getline(monitor_file, line); // The comment line
  std::istringstream line_stream;
  integer counter{0};
  while (gxl::getline_to_stream(monitor_file, line, line_stream)) {
    integer pid;
    line_stream >> pid;
    if (myid != pid) {
      continue;
    }
    integer i, j, k, b;
    line_stream >> b >> i >> j >> k;
    is_h.push_back(i);
    js_h.push_back(j);
    ks_h.push_back(k);
    bs_h.push_back(b);
    ++n_point[b];
    ++counter;
  }
  // copy the indices to GPU
  cudaMalloc(&h_ptr->bs_d, sizeof(integer) * counter);
  cudaMalloc(&h_ptr->is_d, sizeof(integer) * counter);
  cudaMalloc(&h_ptr->js_d, sizeof(integer) * counter);
  cudaMalloc(&h_ptr->ks_d, sizeof(integer) * counter);
  cudaMemcpy(h_ptr->bs_d, bs_h.data(), sizeof(integer) * counter, cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr->is_d, is_h.data(), sizeof(integer) * counter, cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr->js_d, js_h.data(), sizeof(integer) * counter, cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr->ks_d, ks_h.data(), sizeof(integer) * counter, cudaMemcpyHostToDevice);
  n_point_total = counter;
  printf("Process %d has %d monitor points.\n", myid, n_point_total);
  disp.resize(parameter.get_int("n_block"), 0);
  for (integer b = 1; b < n_block; ++b) {
    disp[b] = disp[b - 1] + n_point[b - 1];
  }
  cudaMalloc(&h_ptr->disp, sizeof(integer) * n_block);
  cudaMemcpy(h_ptr->disp, disp.data(), sizeof(integer) * n_block, cudaMemcpyHostToDevice);
  cudaMalloc(&h_ptr->n_point, sizeof(integer) * n_block);
  cudaMemcpy(h_ptr->n_point, n_point.data(), sizeof(integer) * n_block, cudaMemcpyHostToDevice);

  // Create arrays to contain the monitored data.
  mon_var_h.allocate_memory(n_var, output_file, n_point_total, 0);
  h_ptr->data.allocate_memory(n_var, output_file, n_point_total);

  cudaMalloc(&d_ptr, sizeof(DeviceMonitorData));
  cudaMemcpy(d_ptr, h_ptr, sizeof(DeviceMonitorData), cudaMemcpyHostToDevice);

  // create directories and files to contain the monitored data
  const std::filesystem::path out_dir("output/monitor");
  if (!exists(out_dir)) {
    create_directories(out_dir);
  }
  files.resize(n_point_total, nullptr);
  for (integer l = 0; l < n_point_total; ++l) {
    std::string file_name{
        "/monitor_" + std::to_string(bs_h[l]) + '_' + std::to_string(is_h[l]) + '_' + std::to_string(js_h[l]) + '_' +
        std::to_string(ks_h[l]) + ".dat"};
    std::filesystem::path whole_name_path{out_dir.string() + file_name};
    if (!exists(whole_name_path)) {
      files[l] = fopen(whole_name_path.string().c_str(), "a");
      fprintf(files[l], "variables=step,");
      for (const auto &name: var_name_found) {
        fprintf(files[l], "%s,", name.c_str());
      }
      fprintf(files[l], "time\n");
    } else {
      files[l] = fopen(whole_name_path.string().c_str(), "a");
    }
  }
}

std::vector<std::string> Monitor::setup_labels_to_monitor(const Parameter &parameter, const Species &species) {
  auto n_spec{species.n_spec};
  auto &spec_list{species.spec_list};

  auto var_name{parameter.get_string_array("monitor_var")};

  std::vector<integer> bv_idx, sv_idx;
  auto n_found{0};
  std::vector<std::string> var_name_found;
  for (auto name: var_name) {
    name = gxl::to_upper(name);
    if (name == "DENSITY" || name == "RHO") {
      bv_idx.push_back(0);
      var_name_found.emplace_back("Density");
      ++n_found;
    } else if (name == "U") {
      bv_idx.push_back(1);
      var_name_found.emplace_back("U");
      ++n_found;
    } else if (name == "V") {
      bv_idx.push_back(2);
      var_name_found.emplace_back("V");
      ++n_found;
    } else if (name == "W") {
      bv_idx.push_back(3);
      var_name_found.emplace_back("W");
      ++n_found;
    } else if (name == "PRESSURE" || name == "P") {
      bv_idx.push_back(4);
      var_name_found.emplace_back("Pressure");
      ++n_found;
    } else if (name == "TEMPERATURE" || name == "T") {
      bv_idx.push_back(5);
      var_name_found.emplace_back("Temperature");
      ++n_found;
    } else if (n_spec > 0) {
      auto it = spec_list.find(name);
      if (it != spec_list.end()) {
        sv_idx.push_back(it->second);
        var_name_found.emplace_back(name);
        ++n_found;
      }
    } else if (name == "TKE") {
      sv_idx.push_back(n_spec);
      var_name_found.emplace_back("TKE");
      ++n_found;
    } else if (name == "OMEGA") {
      sv_idx.push_back(n_spec + 1);
      var_name_found.emplace_back("Omega");
      ++n_found;
    } else if (name == "MIXTUREFRACTION" || name == "Z") {
      sv_idx.push_back(n_spec + 2);
      var_name_found.emplace_back("Mixture fraction");
      ++n_found;
    } else if (name == "MIXTUREFRACTIONVARIANCE") {
      sv_idx.push_back(n_spec + 3);
      var_name_found.emplace_back("Mixture fraction variance");
      ++n_found;
    } else {
      if (parameter.get_int("myid") == 0) {
        printf("The variable %s is not found in the variable list.\n", name.c_str());
      }
    }
  }

  // copy the index to the class member
  n_bv = (integer) (bv_idx.size());
  n_sv = (integer) (sv_idx.size());
  // The +1 is for physical time
  n_var = n_bv + n_sv + 1;
  h_ptr->n_bv = n_bv;
  h_ptr->n_sv = n_sv;
  h_ptr->n_var = n_var;
  cudaMalloc(&h_ptr->bv_label, sizeof(integer) * n_bv);
  cudaMalloc(&h_ptr->sv_label, sizeof(integer) * n_sv);
  cudaMemcpy(h_ptr->bv_label, bv_idx.data(), sizeof(integer) * n_bv, cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr->sv_label, sv_idx.data(), sizeof(integer) * n_sv, cudaMemcpyHostToDevice);

  return var_name_found;
}

Monitor::~Monitor() {
  for (auto fp: files) {
    fclose(fp);
  }
}

void Monitor::monitor_point(integer step, real physical_time, std::vector<cfd::Field> &field) {
  if (counter_step == 0)
    step_start = step;

  for (integer b = 0; b < n_block; ++b) {
    if (n_point[b] > 0) {
      const auto tpb{128};
      const auto bpg{(n_point[b] - 1) / tpb + 1};
      record_monitor_data<<<bpg, tpb>>>(field[b].d_ptr, d_ptr, b, counter_step % output_file, physical_time);
    }
  }
  ++counter_step;
}

void Monitor::output_data() {
  cudaMemcpy(mon_var_h.data(), h_ptr->data.data(), sizeof(real) * n_var * output_file * n_point_total,
             cudaMemcpyDeviceToHost);

  for (integer p = 0; p < n_point_total; ++p) {
    for (integer l = 0; l < counter_step; ++l) {
      fprintf(files[p], "%d\t", step_start + l);
      for (integer k = 0; k < n_var; ++k) {
        fprintf(files[p], "%e\t", mon_var_h(k, l, p));
      }
      fprintf(files[p], "\n");
    }
  }
  counter_step = 0;
}

__global__ void
record_monitor_data(DZone *zone, DeviceMonitorData *monitor_info, integer blk_id, integer counter_pos,
                    real physical_time) {
  auto idx = (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  if (idx >= monitor_info->n_point[blk_id])
    return;
  auto idx_tot = monitor_info->disp[blk_id] + idx;
  auto i = monitor_info->is_d[idx_tot];
  auto j = monitor_info->js_d[idx_tot];
  auto k = monitor_info->ks_d[idx_tot];

  auto &data = monitor_info->data;
  const auto bv_label = monitor_info->bv_label;
  const auto sv_label = monitor_info->sv_label;
  const auto n_bv{monitor_info->n_bv};
  integer var_counter{0};
  for (integer l = 0; l < n_bv; ++l) {
    data(var_counter, counter_pos, idx_tot) = zone->bv(i, j, k, bv_label[l]);
    ++var_counter;
  }
  for (integer l = 0; l < monitor_info->n_sv; ++l) {
    data(var_counter, counter_pos, idx_tot) = zone->sv(i, j, k, sv_label[l]);
    ++var_counter;
  }
  data(var_counter, counter_pos, idx_tot) = physical_time;
}
} // cfd
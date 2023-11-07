#include "Monitor.cuh"
#include "gxl_lib/MyString.h"
#include <filesystem>
#include "Parallel.h"

namespace cfd {
Monitor::Monitor(const Parameter &parameter, const Species &species) : if_monitor{parameter.get_int("if_monitor")},
                                                                       output_file{parameter.get_int("output_file")} {
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
  n_point = counter;
  printf("Process %d has %d monitor points.\n", myid, n_point);

  // Create arrays to contain the monitored data.
  mon_var_h.allocate_memory(n_var, output_file, n_point, 0);
  h_ptr->data.allocate_memory(n_var, output_file, n_point);

  cudaMalloc(&d_ptr, sizeof(DeviceMonitorData));
  cudaMemcpy(d_ptr, h_ptr, sizeof(DeviceMonitorData), cudaMemcpyHostToDevice);

  // create directories and files to contain the monitored data
  const std::filesystem::path out_dir("output/monitor");
  if (!exists(out_dir)) {
    create_directories(out_dir);
  }
  files.resize(n_point, nullptr);
  for (integer l = 0; l < n_point; ++l) {
    std::string file_name{
        "/monitor_" + std::to_string(bs_h[l]) + '_' + std::to_string(is_h[l]) + '_' + std::to_string(js_h[l]) + '_' +
        std::to_string(ks_h[l]) + ".dat"};
    files[l] = fopen((out_dir.string() + file_name).c_str(), "a");
    fprintf(files[l], "step");
    for (const auto &name: var_name_found) {
      fprintf(files[l], "%s\t", name.c_str());
    }
    fprintf(files[l], "time\n");
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
  h_ptr->n_var = n_var + 1;
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
} // cfd
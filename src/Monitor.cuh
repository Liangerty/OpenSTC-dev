#pragma once

#include "Parameter.h"
#include "ChemData.h"
#include "gxl_lib/Array.hpp"

namespace cfd {

struct Field;

struct DeviceMonitorData {
  integer n_bv{0}, n_sv{0}, n_var{0};
  integer *bv_label{nullptr};
  integer *sv_label{nullptr};
  integer *bs_d{nullptr}, *is_d{nullptr}, *js_d{nullptr}, *ks_d{nullptr};
  integer *disp{nullptr}, *n_point{nullptr};
  ggxl::Array3D<real> data;
};

class Monitor {
public:
  explicit Monitor(const Parameter &parameter, const Species &species);

  void monitor_point(integer step, real physical_time, std::vector<cfd::Field> &field);

  void output_data();

  ~Monitor();

private:
  integer if_monitor{0};
  integer output_file{0};
  integer step_start{0};
  integer counter_step{0};
  integer n_block{0};
  integer n_bv{0}, n_sv{0}, n_var{0};
  std::vector<integer> bs_h, is_h, js_h, ks_h;
  integer n_point_total{0};
  std::vector<integer> n_point;
  std::vector<integer> disp;
  ggxl::Array3DHost<real> mon_var_h;
  DeviceMonitorData *h_ptr, *d_ptr{nullptr};
  std::vector<FILE *> files;
//  ggxl::Array3D<real> mon_var_d, h_ptr_to_mon_var;


private:
  // Utility functions
  std::vector<std::string> setup_labels_to_monitor(const Parameter &parameter, const Species &species);
};

struct DZone;
__global__ void record_monitor_data(DZone *zone, DeviceMonitorData *monitor_info, integer blk_id, integer counter_pos,
                                    real physical_time);

} // cfd

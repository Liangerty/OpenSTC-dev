#pragma once

#include "Parameter.h"
#include "ChemData.h"
#include "gxl_lib/Array.hpp"

namespace cfd {

struct DeviceMonitorData {
  integer n_bv{0}, n_sv{0}, n_var{0};
  integer *bv_label{nullptr};
  integer *sv_label{nullptr};
  integer *bs_d{nullptr}, *is_d{nullptr}, *js_d{nullptr}, *ks_d{nullptr};
  ggxl::Array3D<real> data;
};

class Monitor {
public:
  explicit Monitor(const Parameter &parameter, const Species &species);

  ~Monitor();

private:
  integer if_monitor{0};
  integer output_file{0};
  integer counter_step{0};
  integer n_bv{0}, n_sv{0}, n_var{0};
  std::vector<integer> bs_h, is_h, js_h, ks_h;
  integer n_point{0};
  ggxl::Array3DHost<real> mon_var_h;
  DeviceMonitorData *h_ptr, *d_ptr;
  std::vector<FILE *> files;
//  ggxl::Array3D<real> mon_var_d, h_ptr_to_mon_var;


private:
  // Utility functions
  std::vector<std::string> setup_labels_to_monitor(const Parameter &parameter, const Species &species);
};

} // cfd

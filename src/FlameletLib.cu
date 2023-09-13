#include "FlameletLib.cuh"
#include "gxl_lib/MyString.h"

namespace cfd {
FlameletLib::FlameletLib(const Parameter &parameter) : n_spec{parameter.get_int("n_spec")} {
  switch (parameter.get_int("flamelet_format")) {
    case 0:
      // ACANS format
      read_ACANS_flamelet(parameter);
      break;
    case 1:
    default:
      // FlameMaster format, not implemented yet.
      break;
  }
}

void FlameletLib::read_ACANS_flamelet(const Parameter &parameter) {
  const auto flamelet_file_name{"input_files/"+parameter.get_string("flamelet_file_name")};
  std::ifstream file{flamelet_file_name};
  std::string input;
  std::istringstream line;
  gxl::getline_to_stream(file, input, line);
  line >> n_z >> n_zPrime >> n_chi;

  z.resize(n_z + 1);
  zPrime.resize(n_zPrime + 1, n_z + 1);
  chi_ave.resize(n_chi, n_zPrime + 1, n_z + 1, 0);
  yk.resize(n_spec, n_chi, n_zPrime + 1, n_z + 1, 0);
  for (integer i = 0; i <= n_z; ++i) {
    for (integer j = 0; j <= n_zPrime; ++j) {
      for (integer k = 0; k < n_chi; ++k) {
        line >> z[i] >> zPrime(j, i) >> chi_ave(k, j, i);
        for (integer l = 0; l < n_spec; ++l)
          line >> yk(l, k, j, i);
      }
    }
  }
  file.close();

  file.open("input_files/chemistry/erf.txt");
  gxl::getline_to_stream(file, input, line);
  line >> fzst;
  fz.resize(n_z + 1);
  for (integer i = 0; i <= n_z; ++i) {
    line >> fz[i];
  }
  file.close();

  dz.resize(n_z);
  for (integer i = 0; i < n_z; ++i) {
    dz[i] = z[i + 1] - z[i];
  }

  chi_min.resize(n_zPrime + 1, n_z + 1);
  chi_max.resize(n_zPrime + 1, n_z + 1);
  chi_min_j.resize(n_zPrime + 1, n_z + 1);
  chi_max_j.resize(n_zPrime + 1, n_z + 1);
  for (integer i = 0; i <= n_z; ++i) {
    for (integer j = 0; j <= n_zPrime; ++j) {
      chi_min(j, i) = chi_ave(0, j, i);
      chi_max(j, i) = chi_ave(0, j, i);
      for (integer k = 0; k < n_chi; ++k) {
        if (chi_ave(k, j, i) <= chi_min(j, i)) {
          chi_min(j, i) = chi_ave(k, j, i);
          chi_min_j(j, i) = k;
        }
        if (chi_ave(k, j, i) >= chi_max(j, i)) {
          chi_max(j, i) = chi_ave(k, j, i);
          chi_max_j(j, i) = k;
        }
      }
    }
  }

  ffz.resize(n_z + 1);
  for (integer i = 0; i <= n_z; ++i) {
    ffz[i] = fz[i];
  }

  zi.resize(n_z + 7);
  zi[0] = z[0];
  zi[1] = 1e-6;
  zi[2] = 1e-5;
  zi[3] = 1e-4;
  for (integer i = 1; i < n_z; ++i) {
    zi[i + 3] = z[i];
  }
  zi[n_z + 3] = 1 - 1e-4;
  zi[n_z + 4] = 1 - 1e-5;
  zi[n_z + 5] = 1 - 1e-6;
  zi[n_z + 6] = 1;
}
}// cfd
#include "FlameletLib.cuh"
#include "gxl_lib/MyString.h"
#include "Field.h"

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
  const auto flamelet_file_name{"input_files/" + parameter.get_string("flamelet_file_name")};
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

__device__ void flamelet_source(cfd::DZone *zone, integer i, integer j, integer k, DParameter *param) {
  const auto &m = zone->metric(i, j, k);
  const real xi_x{m(1, 1)}, xi_y{m(1, 2)}, xi_z{m(1, 3)};
  const real eta_x{m(2, 1)}, eta_y{m(2, 2)}, eta_z{m(2, 3)};
  const real zeta_x{m(3, 1)}, zeta_y{m(3, 2)}, zeta_z{m(3, 3)};

  // compute the gradient of mixture fraction
  const auto &sv{zone->sv};
  const integer i_fl{param->i_fl};

  const real mixFrac_x = 0.5 * (xi_x * (sv(i + 1, j, k, i_fl) - sv(i - 1, j, k, i_fl)) +
                                eta_x * (sv(i, j + 1, k, i_fl) - sv(i, j - 1, k, i_fl)) +
                                zeta_x * (sv(i, j, k + 1, i_fl) - sv(i, j, k - 1, i_fl)));
  const real mixFrac_y = 0.5 * (xi_y * (sv(i + 1, j, k, i_fl) - sv(i - 1, j, k, i_fl)) +
                                eta_y * (sv(i, j + 1, k, i_fl) - sv(i, j - 1, k, i_fl)) +
                                zeta_y * (sv(i, j, k + 1, i_fl) - sv(i, j, k - 1, i_fl)));
  const real mixFrac_z = 0.5 * (xi_z * (sv(i + 1, j, k, i_fl) - sv(i - 1, j, k, i_fl)) +
                                eta_z * (sv(i, j + 1, k, i_fl) - sv(i, j - 1, k, i_fl)) +
                                zeta_z * (sv(i, j, k + 1, i_fl) - sv(i, j, k - 1, i_fl)));

  const real prod_mixFrac = 2.0 * zone->mut(i, j, k) / param->Sct
                            * (mixFrac_x * mixFrac_x + mixFrac_y * mixFrac_y + mixFrac_z * mixFrac_z);
  zone->scalar_diss_rate(i, j, k) = 2 * 0.09 * sv(i, j, k, param->n_spec + 1) * sv(i, j, k, i_fl + 1) * param->c_chi;
  const real diss_mixFrac{zone->bv(i, j, k, 0) * zone->scalar_diss_rate(i, j, k)};

  zone->dq(i, j, k, param->n_turb + 5) += zone->jac(i, j, k) * (prod_mixFrac - diss_mixFrac);
}
}// cfd
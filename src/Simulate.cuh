#pragma once

#include "Driver.cuh"
#include <cstdio>
#include "SteadySim.cuh"

namespace cfd{
template<MixtureModel mix_model, TurbulenceMethod turb_method, class turb>
void simulate(Driver<mix_model, turb_method, turb> &driver){
  const auto& parameter{driver.parameter};
  const auto steady{parameter.get_bool("steady")};
  if (steady) {
    steady_simulation<mix_model, turb_method, turb>(driver);
  } else {
    const auto temporal_tag{parameter.get_int("temporal_scheme")};
    switch (temporal_tag) {
      case 11: // For example, if DULUSGS, then add a function to initiate the computation instead of initialize before setting up the scheme as CPU code
        break;
      case 12:
        break;
      default:
        printf("Not implemented");
    }
  }
}
}
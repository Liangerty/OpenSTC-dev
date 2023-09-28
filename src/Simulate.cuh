#pragma once

#include "Driver.cuh"
#include <cstdio>
#include "SteadySim.cuh"
#include "FirstOrderEuler.cuh"

namespace cfd{
template<MixtureModel mix_model, class turb>
void simulate(Driver<mix_model, turb> &driver){
  const auto& parameter{driver.parameter};
  const auto steady{parameter.get_bool("steady")};
  if (steady) {
    steady_simulation<mix_model, turb>(driver);
  } else {
    const auto temporal_tag{parameter.get_int("temporal_scheme")};
    switch (temporal_tag) {
      case 1: // Explicit Euler, only first order time accuracy, should be avoided in most cases.
        first_order_euler<mix_model, turb>(driver);
        break;
      case 2:
        printf("Not implemented");
        break;
      case 3:
      default:
        printf("Not implemented");
    }
  }
}
}
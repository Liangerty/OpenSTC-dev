#pragma once

#include "Driver.cuh"
#include <cstdio>
#include "SteadySim.cuh"
#include "FirstOrderEuler.cuh"
#include "RK.cuh"

namespace cfd {

template<MixtureModel mix_model, class turb>
void simulate(Driver<mix_model, turb> &driver) {
  const auto &parameter{driver.parameter};
  const auto steady{parameter.get_bool("steady")};
  if (steady) {
    // The methods which use only bv do not need to save cv at all, which is the case in steady simulations.
    // In those methods, such as Roe, AUSM..., we do not store the cv variables.
    steady_simulation<mix_model, turb>(driver);
  } else {
    // In my design, the inviscid method may use cv, or only bv.
    // The methods which use only bv, such as Roe, AUSM..., do not need to save cv at all.
    // But we still store the cv array here. Because in methods such as RK, we need multiple stages to update the cv variables.
    // However, in high-order methods, such as the WENO method, we need to store the cv variables.
    // When this happens, the corresponding boundary conditions, data communications would all involve the update of cv.
    const auto inviscid_tag{parameter.get_int("inviscid_scheme")};
    const auto temporal_tag{parameter.get_int("temporal_scheme")};
    if (inviscid_tag < 10) {
      // Inviscid methods which use only bv
      switch (temporal_tag) {
        case 1: // Explicit Euler, only first order time accuracy, should be avoided in most cases.
          first_order_euler<mix_model, turb, reconstruct_bv>(driver);
          break;
        case 2:
          printf("Not implemented");
          break;
        case 3:
        default:
          RK3_bv<mix_model, turb>(driver);
          break;
      }
    } else {
      // We need to reconstruct cv here. Many methods need modification.
      switch (temporal_tag) {
        case 1: // Explicit Euler, only first order time accuracy, should be avoided in most cases.
          first_order_euler<mix_model, turb, reconstruct_cv>(driver);
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
}
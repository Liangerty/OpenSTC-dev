/**
 * @brief The file for declarations of post process procedures
 * @details Every time a new procedure is added, a new label should be added to the Driver.cu file, post_process() function.
 *  The label should be identical in all scenes, from the post_process() method to the 6_post_process file
 * Current available procedures:
 *    0 - compute wall friction and heat flux in 2D, which assumes a j=0 wall
 *    1 -
 */
#pragma once

#include "Parameter.h"
#include "Driver.cuh"

namespace cfd {
class Mesh;

struct Field;
struct DZone;

template<MixtureModel mix_model, class turb>
void post_process(Driver<mix_model, turb> &driver) {
  auto &parameter{driver.parameter};
  static const std::vector<integer> processes{parameter.get_int_array("post_process")};

  for (auto process: processes) {
    switch (process) {
      case 0: // Compute the 2D cf/qw
        wall_friction_heatflux_2d(driver.mesh, driver.field, parameter);
        break;
      default:
        break;
    }
  }
}

// Compute the wall friction and heat flux in 2D. Assume the wall is the j=0 plane
// Procedure 0
void wall_friction_heatflux_2d(const Mesh &mesh, const std::vector<cfd::Field> &field, const Parameter &parameter);

__global__ void wall_friction_heatFlux_2d(cfd::DZone *zone, real *friction, real *heat_flux, real dyn_pressure);
}

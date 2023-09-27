#pragma once

#include "Define.h"
#include <vector>
#include <mpi.h>
#include "Mesh.h"
#include "Field.h"
#include "DParameter.cuh"
#include "FieldOperation.cuh"

namespace cfd {
void data_communication(const Mesh &mesh, std::vector<cfd::Field> &field, const Parameter &parameter, integer step,
                        DParameter *param);

__global__ void inner_communication(DZone *zone, DZone *tar_zone, integer i_face, DParameter *param);

void parallel_communication(const Mesh &mesh, std::vector<cfd::Field> &field, integer step, DParameter *param);

__global__ void setup_data_to_be_sent(DZone *zone, integer i_face, real *data, const DParameter *param);

__global__ void assign_data_received(DZone *zone, integer i_face, const real *data, DParameter *param);
}
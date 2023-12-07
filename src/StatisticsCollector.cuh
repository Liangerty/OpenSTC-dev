#pragma once

#include "Parameter.h"
#include "Mesh.h"
#include "Field.h"
#include "DParameter.cuh"

namespace cfd {

__global__ void collect_basic_variable_statistics(DZone* zone);
__global__ void collect_species_statistics(DZone* zone, DParameter* param);

class StatisticsCollector {
public:
  explicit StatisticsCollector(Parameter &parameter, const Mesh& _mesh, const std::vector<Field> &_field);

  template<MixtureModel mix_model, class turb_method>
  void collect_data(DParameter* param);

private:
  // Basic info
  bool if_collect_statistics{false};
  integer start_iter{0};
  integer counter{0};
  // Data to be bundled
  const Mesh& mesh;
  const std::vector<Field> &field;

};

template<MixtureModel mix_model, class turb_method>
void StatisticsCollector::collect_data(DParameter* param) {
  ++counter;

  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }

  for (integer b = 0; b < mesh.n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    dim3 bpg = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
    collect_basic_variable_statistics<<<bpg, tpb>>>(field[b].d_ptr);
    if constexpr (mix_model!=MixtureModel::Air){
      collect_species_statistics<<<bpg, tpb>>>(field[b].d_ptr, param);
    }
  }
}

} // cfd

#pragma once

#include "Define.h"
#include "FieldIO.h"
#include "BoundaryIO.h"

namespace cfd {
template<MixtureModel mix_model, TurbulenceMethod turb_method>
struct IOManager {
  FieldIO<mix_model, turb_method, OutputTimeChoice::Instance> field_io;
  BoundaryIO<mix_model, turb_method, OutputTimeChoice::Instance> boundary_io;

  explicit IOManager(integer _myid, const Mesh &_mesh, std::vector<Field> &_field, const Parameter &_parameter,
                     const Species &spec, integer ngg_out);

  void print_field(integer step, const Parameter &parameter);
};

template<MixtureModel mix_model, TurbulenceMethod turb_method>
void IOManager<mix_model, turb_method>::print_field(integer step, const Parameter &parameter) {
  field_io.print_field(step);
  boundary_io.print_boundary();

  if constexpr (mix_model == MixtureModel::FL) {
    std::ofstream out("output/message/flamelet_step.txt", std::ios::trunc);
    out << parameter.get_int("n_fl_step") << std::endl;
    out.close();
  }
}

template<MixtureModel mix_model, TurbulenceMethod turb_method>
IOManager<mix_model, turb_method>::IOManager(integer _myid, const Mesh &_mesh, std::vector<Field> &_field,
                                             const Parameter &_parameter, const Species &spec, integer ngg_out):
    field_io(_myid, _mesh, _field, _parameter, spec, ngg_out), boundary_io(_parameter, _mesh, spec, _field) {

}

template<MixtureModel mix_model, TurbulenceMethod turb_method>
struct TimeSeriesIOManager {
  FieldIO<mix_model, turb_method, OutputTimeChoice::TimeSeries> field_io;
  BoundaryIO<mix_model, turb_method, OutputTimeChoice::TimeSeries> boundary_io;
};
}
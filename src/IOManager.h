#pragma once

#include "Define.h"
#include "FieldIO.h"
#include "BoundaryIO.h"

namespace cfd {
template<MixtureModel mix_model, class turb>
struct IOManager {
  FieldIO<mix_model, turb, OutputTimeChoice::Instance> field_io;
  BoundaryIO<mix_model, turb, OutputTimeChoice::Instance> boundary_io;

  explicit IOManager(integer _myid, const Mesh &_mesh, std::vector<Field> &_field, const Parameter &_parameter,
                     const Species &spec, integer ngg_out);

  void print_field(integer step, const Parameter &parameter, real physical_time = 0);
};

template<MixtureModel mix_model, class turb>
void IOManager<mix_model, turb>::print_field(integer step, const Parameter &parameter, real physical_time) {
  field_io.print_field(step, physical_time);
  boundary_io.print_boundary();

  if constexpr (mix_model == MixtureModel::FL) {
    std::ofstream out("output/message/flamelet_step.txt", std::ios::trunc);
    out << parameter.get_int("n_fl_step") << std::endl;
    out.close();
  }
}

template<MixtureModel mix_model, class turb>
IOManager<mix_model, turb>::IOManager(integer _myid, const Mesh &_mesh, std::vector<Field> &_field,
                                      const Parameter &_parameter, const Species &spec, integer ngg_out):
    field_io(_myid, _mesh, _field, _parameter, spec, ngg_out), boundary_io(_parameter, _mesh, spec, _field) {

}

template<MixtureModel mix_model, class turb_method>
struct TimeSeriesIOManager {
  FieldIO<mix_model, turb_method, OutputTimeChoice::TimeSeries> field_io;
//  BoundaryIO<mix_model, turb_method, OutputTimeChoice::TimeSeries> boundary_io;

  explicit TimeSeriesIOManager(integer _myid, const Mesh &_mesh, std::vector<Field> &_field,
                               const Parameter &_parameter,
                               const Species &spec, integer ngg_out);

  void print_field(integer step, const Parameter &parameter, real physical_time);
};

template<MixtureModel mix_model, class turb_method>
TimeSeriesIOManager<mix_model, turb_method>::TimeSeriesIOManager(integer _myid, const Mesh &_mesh,
                                                                 std::vector<Field> &_field,
                                                                 const Parameter &_parameter, const Species &spec,
                                                                 integer ngg_out):
    field_io(_myid, _mesh, _field, _parameter, spec, ngg_out)/*, boundary_io(_parameter, _mesh, spec, _field)*/ {

}

template<MixtureModel mix_model, class turb_method>
void
TimeSeriesIOManager<mix_model, turb_method>::print_field(integer step, const Parameter &parameter, real physical_time) {
  field_io.print_field(step, physical_time);
}

}
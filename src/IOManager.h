#pragma once
#include "Define.h"
#include "FieldIO.h"
#include "BoundaryIO.h"

namespace cfd{
template<MixtureModel mix_model, TurbMethod turb_method>
struct IOManager{
  FieldIO<mix_model, turb_method, OutputTimeChoice::Instance> field_io;
  BoundaryIO<mix_model, turb_method, OutputTimeChoice::Instance> boundary_io;
};

template<MixtureModel mix_model, TurbMethod turb_method>
struct TimeSeriesIOManager{
  FieldIO<mix_model, turb_method, OutputTimeChoice::TimeSeries> field_io;
  BoundaryIO<mix_model, turb_method, OutputTimeChoice::TimeSeries> boundary_io;
};
}
//#pragma once
//
//#include "Driver.cuh"
//#include "FlameletLib.cuh"
//
//namespace cfd{
////template<TurbulenceMethod turb_method>
////struct Driver<MixtureModel::FL, turb_method>{
////  Driver(Parameter &parameter, Mesh &mesh_);
////
////  void initialize_computation();
////  void simulate();
////
////  integer myid = 0;
////  gxl::Time time;
////  const Mesh &mesh;
////  const Parameter &parameter;
////  Species spec;
////  FlameletLib flameletLib;
////  std::vector<cfd::Field> field;
////  DParameter *param = nullptr; // The parameters used for GPU simulation, data are stored on GPU while the pointer is on CPU
////  DBoundCond bound_cond;  // Boundary conditions
////  std::array<real, 4> res{1, 1, 1, 1};
////  std::array<real, 4> res_scale{1, 1, 1, 1};
////};
//
////extern template<> struct Driver<MixtureModel::FL, TurbulenceMethod::RANS>;
//}
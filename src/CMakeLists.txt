# There are several options need to be set here
# 1. "CMAKE_CUDA_ARCHITECTURES", the CUDA architecture should be specified
#     For example, for Nvidia Quadro GP100, the value should be 60, while for GV100, 70
# 2. "MAX_SPEC_NUMBER" if mixture, the number of species should not beyond 55, which is set by allowing the computation of CH4 in Grimech mechanism.
#     If larger mechanism is to be used, the variable should be modified to be larger or equal to the species number.
#     However, the correctness, if the resource is available for the corresponding kernel may need to be tested.

list(APPEND src_file
  AusmP.cuh AusmP.cu AWENO.cuh AWENO.cu
  BoundCond.h BoundCond.cpp BoundCond.cu BoundCond.cuh BoundaryIO.h
  ChemData.h ChemData.cpp Constants.h
  DataCommunication.cuh DataCommunication.cu Define.h DParameter.cu DParameter.cuh DPLUR.cuh DPLUR.cu Driver.cu Driver.cuh
  Element.h Element.cpp
  Field.h Field.cu FieldOperation.cuh FieldOperation.cu FiniteRateChem.cu FiniteRateChem.cuh FieldIO.h FirstOrderEuler.cuh
  FlameletDriver.cuh FlameletDriver.cu FlameletLib.cu FlameletLib.cuh
  HLLC.cuh HLLC.cu
  ImplicitTreatmentHPP.cuh Initialize.cuh Initialize.cu InviscidScheme.cuh InviscidScheme.cu IOManager.h IOBase.h
  Limiter.cuh LaxFriedrichs.cuh LaxFriedrichs.cu
  main.cu Mesh.h Mesh.cpp MPIIO.hpp MPIIO.cu MixtureFraction.cpp MixtureFraction.h Monitor.cuh Monitor.cu
  Output.hpp
  Parallel.h Parallel.cpp Parallel.cu Parameter.h Parameter.cpp PostProcess.h PostProcess.cu
  Reconstruction.cu Reconstruction.cuh Residual.cu Residual.cuh Roe.cuh Roe.cu RK.cuh
  SchemeSelector.cuh SourceTerm.cuh SST.cuh SST.cu SteadySim.cuh Simulate.cuh StatisticsCollector.cu StatisticsCollector.cuh
  Transport.cuh Transport.cu TimeAdvanceFunc.cu TimeAdvanceFunc.cuh Thermo.cuh Thermo.cu TurbMethod.hpp
  ViscousScheme.cuh
  WallDistance.cuh

  gxl_lib/Array.hpp
  gxl_lib/MyString.cpp
  gxl_lib/Time.cpp gxl_lib/Time.h
  gxl_lib/MyAtomic.cuh
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

cmake_policy(SET CMP0074 NEW)
find_package(MPI REQUIRED)

add_executable(openstc ${src_file})

set_target_properties(openstc PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(openstc
  PROPERTIES
  CXX_STANDARD 20 # A newer version CMake 3.25.2 would support this
  CXX_EXTENSIONS OFF
  CXX_STANDARD_REQUIRED ON
  #  "$<$<COMPILE_LANGUAGE:CXX>:${flags_release}>"
)

target_include_directories(openstc
  PUBLIC
  ${CMAKE_SOURCE_DIR}/depends/include/
)

target_link_directories(openstc
  PUBLIC#PRIVATE
  "$<$<CONFIG:Debug>:${CMAKE_SOURCE_DIR}/depends/lib/debug>"
  "$<$<CONFIG:Release>:${CMAKE_SOURCE_DIR}/depends/lib/release>"
)

#target_compile_options(2_CLionGPU
#  PUBLIC#private
#  ${flags}
#  "$<$<COMPILE_LANGUAGE:CXX>:${flags_release}>"
#  )

target_link_libraries(openstc
  PUBLIC#PRIVATE
  "$<$<CONFIG:Debug>:fmtd>"
  "$<$<CONFIG:Release>:fmt>"
  MPI::MPI_CXX
)

#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#define __host__
#endif

using integer = int;
using real = double;
using uint = unsigned int;

enum class TurbMethod{
  Laminar,
  RANS,
  LES,
//  ILES,
//  DNS
};

enum class MixtureModel{
  Air,
  Mixture,  // Species mixing
  FR,       // Finite Rate
  FL,       // Flamelet Model
};

enum class OutputTimeChoice{
  Instance,   // Output the instant values, which would overwrite its previous values
  TimeSeries, // Output the values as a time series, which would create new files with time stamp
};

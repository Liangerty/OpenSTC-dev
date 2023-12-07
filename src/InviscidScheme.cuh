#pragma once

#include "Define.h"

namespace cfd {
class Block;
struct DZone;
struct DParameter;
class Parameter;

template<MixtureModel mix_model>
void compute_convective_term_pv(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var, const Parameter &parameter);
template<MixtureModel mix_model>
__global__ void compute_convective_term_pv_1D(cfd::DZone *zone, integer direction, integer max_extent, DParameter *param);

template<MixtureModel mix_model>
void compute_convective_term_aweno(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var, const Parameter &parameter);

template<MixtureModel mix_model>
__global__ void compute_convective_term_aweno_1D(cfd::DZone *zone, integer direction, integer max_extent, DParameter *param);
} // cfd

#pragma once

#include "Define.h"
#include "gxl_lib/Matrix.hpp"
#include "gxl_lib/Array.hpp"
#include "BoundCond.h"
#include "ChemData.h"

namespace cfd {

struct MixtureFraction {
  integer n_spec{0};
  real beta_f{0}, beta_o{0};
  real beta_diff{0};
  real z_st{0}, f_zst{0};
  const gxl::MatrixDyn<integer>& elem_comp;
  const std::vector<real>& mw;

  explicit MixtureFraction(Inflow& fuel, Inflow& oxidizer, const Species &chem_data);

  virtual real compute_mixture_fraction(std::vector<real>& yk)=0;
};

class BilgerH:public MixtureFraction{
  real nuh_mwh{0},half_nuo_mwo{0};
  integer elem_label[2]{0,1};
public:
  explicit BilgerH(Inflow& fuel, Inflow& oxidizer, const Species &chem_data, integer myid=0);

  real compute_mixture_fraction(std::vector<real> &yk) override;
private:
  [[nodiscard]] real compute_coupling_function(real z_h, real z_o) const;
};

class BilgerCH:public MixtureFraction{
  real nuc_mwc{0},nuh_mwh{0},half_nuo_mwo{0};
  integer elem_label[3]{0,1,2};
public:
  explicit BilgerCH(Inflow& fuel, Inflow& oxidizer, const Species &chem_data, integer myid=0);

  real compute_mixture_fraction(std::vector<real> &yk) override;
private:
  [[nodiscard]] real compute_coupling_function(real z_c, real z_h, real z_o) const;
};


} // cfd

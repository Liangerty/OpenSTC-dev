#pragma once

#include "Parameter.h"
#include "Define.h"
#include "gxl_lib/Matrix.hpp"
#include "gxl_lib/Array.hpp"

namespace cfd {
struct Species;
struct Reaction;
struct FlameletLib;

struct DParameter {
  DParameter() = default;

  explicit DParameter(cfd::Parameter &parameter, Species &species, Reaction *reaction,
                      FlameletLib *flamelet_lib = nullptr);

  integer myid = 0;   // The process id of this process

  // Number of equations and variables
  integer n_var=0;                    // The number of variables in the conservative variable
  integer n_scalar = 0;               // The number of scalar variables
  integer n_scalar_transported = 0;   // The number of scalar variables in the conservative equation, this is only different from n_scalar when we use flamelet model
  integer n_spec = 0;                 // The number of species
  integer i_fl = 0;                   // The index of flamelet variable in the scalar variable
  integer i_fl_cv = 0;                // The index of flamelet variable in the conservative variable
  integer i_turb_cv = 0;              // The index of turbulent variable in the conservative variable

  integer inviscid_scheme = 0;  // The tag for inviscid scheme. 3 - AUSM+
  integer reconstruction = 2; // The reconstruction method for inviscid flux computation
  integer limiter = 0;  // The tag for limiter method
  integer viscous_scheme = 0; // The tag for viscous scheme. 0 - Inviscid, 2 - 2nd order central discretization

  integer rans_model = 0;  // The tag for RANS model. 0 - Laminar, 1 - SA, 2 - SST
  integer turb_implicit = 1;    // If we implicitly treat the turbulent source term. By default, implicitly treat(1), else, 0(explicit)
  integer compressibility_correction = 0; // Which compressibility correction to be used. 0 - No compressibility correction, 1 - Wilcox's correction, 2 - Sarkar's correction, 3 - Zeman's correction

  integer chemSrcMethod = 0;  // For finite rate chemistry, we need to know how to implicitly treat the chemical source

  integer n_reac = 0;
  real Pr = 0.72;
  real cfl = 1;

  real *mw = nullptr;
  ggxl::MatrixDyn<real> high_temp_coeff, low_temp_coeff;
  real *t_low = nullptr, *t_mid = nullptr, *t_high = nullptr;

  // Transport properties
  real *LJ_potent_inv = nullptr;
  real *vis_coeff = nullptr;
  ggxl::MatrixDyn<real> WjDivWi_to_One4th;
  ggxl::MatrixDyn<real> sqrt_WiDivWjPl1Mul8;
  ggxl::MatrixDyn<real> binary_diffusivity_coeff;
  ggxl::MatrixDyn<real> kb_over_eps_jk; // Used to compute reduced temperature for diffusion coefficients
  bool gradPInDiffusionFlux = false;

  real Sc = 0.9;
  real Prt = 0.9;
  real Sct = 0.9;
  integer *reac_type = nullptr;
  ggxl::MatrixDyn<integer> stoi_f, stoi_b;
  integer *reac_order = nullptr;
  real *A = nullptr, *b = nullptr, *Ea = nullptr;
  real *A2 = nullptr, *b2 = nullptr, *Ea2 = nullptr;
  ggxl::MatrixDyn<real> third_body_coeff;
  real *troe_alpha = nullptr, *troe_t3 = nullptr, *troe_t1 = nullptr, *troe_t2 = nullptr;

  // Flamelet library info
  integer n_z = 0, n_zPrime = 0, n_chi = 0;
  integer n_fl_step = 0;
  real *mix_frac = nullptr;
  ggxl::MatrixDyn<real> zPrime, chi_min, chi_max;
  ggxl::MatrixDyn<integer> chi_min_j, chi_max_j;
  ggxl::Array3D<real> chi_ave;
  ggxl::VectorField3D<real> yk_lib;
  real c_chi{1.0};

private:
  struct LimitFlow {
    // ll for lower limit, ul for upper limit.
    static constexpr integer max_n_var = 5 + 2;
    real ll[max_n_var];
    real ul[max_n_var];
    real sv_inf[MAX_SPEC_NUMBER + 4];
  };

public:
  LimitFlow limit_flow{};
};
}

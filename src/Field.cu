#include "Field.h"
#include "BoundCond.h"

cfd::Field::Field(Parameter &parameter, const Block &block_in) : block(block_in) {
  const integer mx{block.mx}, my{block.my}, mz{block.mz}, ngg{block.ngg};
  // Let us re-compute the number of variables to be solved here.
  n_var = 5;
  // The variable "n_scalar_transported" is the number of scalar variables to be transported.
  integer n_scalar_transported{0};
  integer n_other_var{1}; // Default, mach number
  // The variable "n_scalar" is the number of scalar variables in total, including those not transported.
  // This is different from the variable "n_scalar_transported" only when the flamelet model is used.
  integer n_scalar{0};
  // turbulent variable in sv array is always located after mass fractions, thus its label is always n_spec;
  // however, for conservative array, it may depend on whether the employed method is flamelet or finite rate.
  // This label, "i_turb_cv", is introduced to label the index of the first turbulent variable in the conservative variable array.
  integer i_turb_cv{5}, i_fl_cv{0};

  if (parameter.get_int("species") == 1) {
    n_scalar += parameter.get_int("n_spec");
    if (parameter.get_int("reaction") != 2) {
      // Mixture / Finite rate chemistry
      n_scalar_transported += parameter.get_int("n_spec");
      i_turb_cv += parameter.get_int("n_spec");
    } else {
      // Flamelet model
      n_scalar_transported += 2; // the mixture fraction and the variance of mixture fraction
      n_scalar += 2;
//      i_turb_cv = 5;
      i_fl_cv = 5 + parameter.get_int("n_turb");
      ++n_other_var; // scalar dissipation rate
    }
  } else if (parameter.get_int("species") == 2) {
    n_scalar += parameter.get_int("n_spec");
    if (parameter.get_int("reaction") != 2) {
      // Mixture with mixture fraction and variance solved.
      n_scalar_transported += parameter.get_int("n_spec") + 2;
      n_scalar += 2;
      i_turb_cv += parameter.get_int("n_spec");
      i_fl_cv = i_turb_cv + parameter.get_int("n_turb");
      ++n_other_var; // scalar dissipation rate
    } else {
      // Flamelet model
      n_scalar_transported += 2; // the mixture fraction and the variance of mixture fraction
      n_scalar += 2;
//      i_turb_cv = 5;
      i_fl_cv = 5 + parameter.get_int("n_turb");
      ++n_other_var; // scalar dissipation rate
    }
  }
  if (parameter.get_bool("turbulence")) {
    // turbulence simulation
    if (parameter.get_int("turbulence_method") == 1) {
      // RANS
      n_scalar_transported += parameter.get_int("n_turb");
      n_scalar += parameter.get_int("n_turb");
//      i_turb_cv = 5;
      ++n_other_var; // mut
    }
  }
  n_var += n_scalar_transported;
  parameter.update_parameter("n_var", n_var);
  parameter.update_parameter("n_scalar", n_scalar);
  parameter.update_parameter("n_scalar_transported", n_scalar_transported);
  parameter.update_parameter("i_turb_cv", i_turb_cv);
  parameter.update_parameter("i_fl", parameter.get_int("n_turb") + parameter.get_int("n_spec"));
  parameter.update_parameter("i_fl_cv", i_fl_cv);

  // Acquire memory for variable arrays
  bv.resize(mx, my, mz, 6, ngg);
  sv.resize(mx, my, mz, n_scalar, ngg);
  ov.resize(mx, my, mz, n_other_var, ngg);

  if (parameter.get_bool("if_collect_statistics")) {
    // If we need to collect the statistics, we need to allocate memory for the data.
    sum12Moment.resize(mx, my, mz, 12, 0);
    sum34Moment.resize(mx, my, mz, 12, 0);
    sumReynoldsShearPart.resize(mx, my, mz, 3, 0);
    if (parameter.get_int("species") != 0){
      sumYk.resize(mx,my,mz,parameter.get_int("n_spec"),0);
    }
  }
}

void cfd::Field::initialize_basic_variables(const Parameter &parameter, const std::vector<Inflow> &inflows,
                                            const std::vector<real> &xs, const std::vector<real> &xe,
                                            const std::vector<real> &ys, const std::vector<real> &ye,
                                            const std::vector<real> &zs, const std::vector<real> &ze) {
  const auto n = inflows.size();
  std::vector<real> rho(n, 0), u(n, 0), v(n, 0), w(n, 0), p(n, 0), T(n, 0);
  const integer n_scalar = parameter.get_int("n_scalar");
  gxl::MatrixDyn<real> scalar_inflow{static_cast<int>(n), n_scalar};

  for (size_t i = 0; i < inflows.size(); ++i) {
    std::tie(rho[i], u[i], v[i], w[i], p[i], T[i]) = inflows[i].var_info();
  }
  for (size_t i = 0; i < inflows.size(); ++i) {
    auto sv_this = inflows[i].sv;
    for (int l = 0; l < n_scalar; ++l) {
      scalar_inflow(static_cast<int>(i), l) = sv_this[l];
    }
  }

  const int ngg{block.ngg};
  for (int i = -ngg; i < block.mx + ngg; ++i) {
    for (int j = -ngg; j < block.my + ngg; ++j) {
      for (int k = -ngg; k < block.mz + ngg; ++k) {
        size_t i_init{0};
        if (inflows.size() > 1) {
          for (size_t l = 0; l < inflows.size() - 1; ++l) {
            if (block.x(i, j, k) >= xs[l] && block.x(i, j, k) <= xe[l]
                && block.y(i, j, k) >= ys[l] && block.y(i, j, k) <= ye[l]
                && block.z(i, j, k) >= zs[l] && block.z(i, j, k) <= ze[l]) {
              i_init = l + 1;
              break;
            }
          }
        }
        bv(i, j, k, 0) = rho[i_init];
        bv(i, j, k, 1) = u[i_init];
        bv(i, j, k, 2) = v[i_init];
        bv(i, j, k, 3) = w[i_init];
        bv(i, j, k, 4) = p[i_init];
        bv(i, j, k, 5) = T[i_init];
        for (integer l = 0; l < n_scalar; ++l) {
          sv(i, j, k, l) = scalar_inflow(static_cast<int>(i_init), l);
        }
      }
    }
  }
}

void cfd::Field::setup_device_memory(const Parameter &parameter) {
  h_ptr = new DZone;
  const auto mx{block.mx}, my{block.my}, mz{block.mz}, ngg{block.ngg};
  h_ptr->mx = mx, h_ptr->my = my, h_ptr->mz = mz, h_ptr->ngg = ngg;

  h_ptr->x.allocate_memory(mx, my, mz, ngg);
  cudaMemcpy(h_ptr->x.data(), block.x.data(), sizeof(real) * h_ptr->x.size(), cudaMemcpyHostToDevice);
  h_ptr->y.allocate_memory(mx, my, mz, ngg);
  cudaMemcpy(h_ptr->y.data(), block.y.data(), sizeof(real) * h_ptr->y.size(), cudaMemcpyHostToDevice);
  h_ptr->z.allocate_memory(mx, my, mz, ngg);
  cudaMemcpy(h_ptr->z.data(), block.z.data(), sizeof(real) * h_ptr->z.size(), cudaMemcpyHostToDevice);

  auto n_bound{block.boundary.size()};
  auto n_inner{block.inner_face.size()};
  auto n_par{block.parallel_face.size()};
  auto mem_sz = sizeof(Boundary) * n_bound;
  cudaMalloc(&h_ptr->boundary, mem_sz);
  cudaMemcpy(h_ptr->boundary, block.boundary.data(), mem_sz, cudaMemcpyHostToDevice);
  mem_sz = sizeof(InnerFace) * n_inner;
  cudaMalloc(&h_ptr->innerface, mem_sz);
  cudaMemcpy(h_ptr->innerface, block.inner_face.data(), mem_sz, cudaMemcpyHostToDevice);
  mem_sz = sizeof(ParallelFace) * n_par;
  cudaMalloc(&h_ptr->parface, mem_sz);
  cudaMemcpy(h_ptr->parface, block.parallel_face.data(), mem_sz, cudaMemcpyHostToDevice);

  h_ptr->jac.allocate_memory(mx, my, mz, ngg);
  cudaMemcpy(h_ptr->jac.data(), block.jacobian.data(), sizeof(real) * h_ptr->jac.size(), cudaMemcpyHostToDevice);
  h_ptr->metric.allocate_memory(mx, my, mz, ngg);
  cudaMemcpy(h_ptr->metric.data(), block.metric.data(), sizeof(gxl::Matrix<real, 3, 3, 1>) * h_ptr->metric.size(),
             cudaMemcpyHostToDevice);

  h_ptr->cv.allocate_memory(mx, my, mz, n_var, ngg);
  h_ptr->bv.allocate_memory(mx, my, mz, 6, ngg);
  cudaMemcpy(h_ptr->bv.data(), bv.data(), sizeof(real) * h_ptr->bv.size() * 6, cudaMemcpyHostToDevice);
  h_ptr->bv_last.allocate_memory(mx, my, mz, 4, 0);
  h_ptr->vel.allocate_memory(mx, my, mz, ngg);
  h_ptr->acoustic_speed.allocate_memory(mx, my, mz, ngg);
  h_ptr->mach.allocate_memory(mx, my, mz, ngg);
  h_ptr->mul.allocate_memory(mx, my, mz, ngg);
  h_ptr->thermal_conductivity.allocate_memory(mx, my, mz, ngg);

  const auto n_spec{parameter.get_int("n_spec")};
  const auto n_scalar = parameter.get_int("n_scalar");
  h_ptr->sv.allocate_memory(mx, my, mz, n_scalar, ngg);
  cudaMemcpy(h_ptr->sv.data(), sv.data(), sizeof(real) * h_ptr->sv.size() * n_scalar, cudaMemcpyHostToDevice);
  h_ptr->rho_D.allocate_memory(mx, my, mz, n_spec, ngg);
  if (n_spec > 0) {
    h_ptr->gamma.allocate_memory(mx, my, mz, ngg);
    h_ptr->cp.allocate_memory(mx, my, mz, ngg);
    if (parameter.get_int("reaction") == 1) {
      // Finite rate chemistry
      if (const integer chemSrcMethod = parameter.get_int("chemSrcMethod");chemSrcMethod == 1) {
        // EPI
        h_ptr->chem_src_jac.allocate_memory(mx, my, mz, n_spec * n_spec, 0);
      } else if (chemSrcMethod == 2) {
        // DA
        h_ptr->chem_src_jac.allocate_memory(mx, my, mz, n_spec, 0);
      }
    } else if (parameter.get_int("reaction") == 2 || parameter.get_int("species") == 2) {
      // Flamelet model
      h_ptr->scalar_diss_rate.allocate_memory(mx, my, mz, ngg);
      // Maybe we can also implicitly treat the source term here.
    }
  }
  if (parameter.get_int("turbulence_method") == 1) {
    // RANS method
    h_ptr->mut.allocate_memory(mx, my, mz, ngg);
    h_ptr->turb_therm_cond.allocate_memory(mx, my, mz, ngg);
    if (parameter.get_int("RANS_model") == 2) {
      // SST
      h_ptr->wall_distance.allocate_memory(mx, my, mz, ngg);
      if (parameter.get_int("turb_implicit") == 1) {
        h_ptr->turb_src_jac.allocate_memory(mx, my, mz, 2, 0);
      }
    }
  }

  h_ptr->dq.allocate_memory(mx, my, mz, n_var, 0);
  h_ptr->inv_spectr_rad.allocate_memory(mx, my, mz, 0);
  h_ptr->visc_spectr_rad.allocate_memory(mx, my, mz, 0);
  if (parameter.get_int("implicit_method") == 1) { // DPLUR
    // If DPLUR type, when computing the products of convective jacobian and dq, we need 1 layer of ghost grids whose dq=0.
    // Except those inner or parallel communication faces, they need to get the dq from neighbor blocks.
    h_ptr->dq.allocate_memory(mx, my, mz, n_var, 1);
    h_ptr->dq0.allocate_memory(mx, my, mz, n_var, 1);
    h_ptr->dqk.allocate_memory(mx, my, mz, n_var, 1);
    h_ptr->inv_spectr_rad.allocate_memory(mx, my, mz, 1);
    h_ptr->visc_spectr_rad.allocate_memory(mx, my, mz, 1);
  }
//  if (parameter.get_bool("steady")) { // steady simulation
  h_ptr->dt_local.allocate_memory(mx, my, mz, 0);
  h_ptr->unphysical.allocate_memory(mx, my, mz, 0);
//  }
  if (parameter.get_int("inviscid_scheme") == 2) {
    // Roe scheme
    h_ptr->entropy_fix_delta.allocate_memory(mx, my, mz, 1);
  }

  if (!parameter.get_bool("steady")) {
    // unsteady simulation
    if (parameter.get_int("temporal_scheme") == 3) {
      // rk scheme
      h_ptr->qn.allocate_memory(mx, my, mz, n_var, ngg);
    }
  }

  if (parameter.get_bool("if_collect_statistics")) {
    // If we need to collect the statistics, we need to allocate memory for the data.
    h_ptr->sum12Moment.allocate_memory(mx, my, mz, 12, 0);
    h_ptr->sum34Moment.allocate_memory(mx, my, mz, 12, 0);
    h_ptr->sumReynoldsShearPart.allocate_memory(mx, my, mz, 3, 0);
    if (parameter.get_int("species") != 0){
      h_ptr->sumYk.allocate_memory(mx,my,mz,parameter.get_int("n_spec"),0);
    }
  }

  cudaMalloc(&d_ptr, sizeof(DZone));
  cudaMemcpy(d_ptr, h_ptr, sizeof(DZone), cudaMemcpyHostToDevice);
}

void cfd::Field::copy_data_from_device(const Parameter &parameter) {
  const auto size = (block.mx + 2 * block.ngg) * (block.my + 2 * block.ngg) * (block.mz + 2 * block.ngg);

  cudaMemcpy(bv.data(), h_ptr->bv.data(), 6 * size * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(ov.data(), h_ptr->mach.data(), size * sizeof(real), cudaMemcpyDeviceToHost);
  if (parameter.get_int("turbulence_method") == 1) {
    cudaMemcpy(ov[1], h_ptr->mut.data(), size * sizeof(real), cudaMemcpyDeviceToHost);
  }
  cudaMemcpy(sv.data(), h_ptr->sv.data(), parameter.get_int("n_scalar") * size * sizeof(real), cudaMemcpyDeviceToHost);
  if (parameter.get_int("reaction") == 2 || parameter.get_int("species") == 2) {
    cudaMemcpy(ov[2], h_ptr->scalar_diss_rate.data(), size * sizeof(real), cudaMemcpyDeviceToHost);
  }
}

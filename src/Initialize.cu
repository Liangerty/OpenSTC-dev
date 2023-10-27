#include "Initialize.cuh"
#include "MixtureFraction.h"

namespace cfd {
void initialize_from_start(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, Species &species) {
  // First, find out how many groups of initial conditions are needed.
  const integer tot_group{parameter.get_int("groups_init")};
  std::vector<Inflow> groups_inflow;
  const std::string default_init = parameter.get_string("default_init");
  Inflow default_inflow(default_init, species, parameter);
  groups_inflow.push_back(default_inflow);

  std::vector<real> xs{}, xe{}, ys{}, ye{}, zs{}, ze{};
  if (tot_group > 1) {
    for (integer l = 0; l < tot_group - 1; ++l) {
      auto patch_struct_name = "init_cond_" + std::to_string(l);
      auto &patch_cond = parameter.get_struct(patch_struct_name);
      xs.push_back(std::get<real>(patch_cond.at("x0")));
      xe.push_back(std::get<real>(patch_cond.at("x1")));
      ys.push_back(std::get<real>(patch_cond.at("y0")));
      ye.push_back(std::get<real>(patch_cond.at("y1")));
      zs.push_back(std::get<real>(patch_cond.at("z0")));
      ze.push_back(std::get<real>(patch_cond.at("z1")));
      //groups_inflow.emplace_back(patch_struct_name, species, parameter);
      if (patch_cond.find("name") != patch_cond.cend()) {
        auto name = std::get<std::string>(patch_cond.at("name"));
        groups_inflow.emplace_back(name, species, parameter);
      } else {
        groups_inflow.emplace_back(patch_struct_name, species, parameter);
      }
    }
  }

  // Start to initialize
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    field[blk].initialize_basic_variables(parameter, groups_inflow, xs, xe, ys, ye, zs, ze);
  }


  if (parameter.get_int("myid") == 0) {
    printf("Flowfield is initialized from given inflow conditions.\n");
    std::ofstream history("history.dat", std::ios::trunc);
    history << "step\terror_max\n";
    history.close();
  }
}

void initialize_spec_from_inflow(cfd::Parameter &parameter, const cfd::Mesh &mesh,
                                 std::vector<Field> &field, Species &species) {
  // This can also be implemented like the from_start one, which can have patch.
  // But currently, for easy to implement, just initialize the whole flowfield to the inflow composition,
  // which means that other species would have to be computed from boundary conditions.
  // If the need for initialize species in groups is strong,
  // then we implement it just by copying the previous function "initialize_from_start",
  // which should be easy.
  const std::string default_init = parameter.get_string("default_init");
  Inflow inflow(default_init, species, parameter);
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    const integer mx{mesh[blk].mx}, my{mesh[blk].my}, mz{mesh[blk].mz};
    const auto n_spec = parameter.get_int("n_spec");
    auto mass_frac = inflow.sv;
    auto &yk = field[blk].sv;
    for (int k = 0; k < mz; ++k) {
      for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
          for (int l = 0; l < n_spec; ++l) {
            yk(i, j, k, l) = mass_frac[l];
          }
        }
      }
    }
  }
  if (parameter.get_int("myid") == 0) {
    printf("Compute from single species result. The species field is initialized with freestream.\n");
  }
  // If flamelet model, the mixture fraction should also be initialized
  if (parameter.get_int("reaction") == 2) {
    const integer i_fl{parameter.get_int("i_fl")};
    for (int blk = 0; blk < mesh.n_block; ++blk) {
      const integer mx{mesh[blk].mx}, my{mesh[blk].my}, mz{mesh[blk].mz};
      auto sv_in = inflow.sv;
      auto &sv = field[blk].sv;
      for (int k = 0; k < mz; ++k) {
        for (int j = 0; j < my; ++j) {
          for (int i = 0; i < mx; ++i) {
            sv(i, j, k, i_fl) = sv_in[i_fl];
            sv(i, j, k, i_fl + 1) = 0;
          }
        }
      }
    }
  }
}

void initialize_turb_from_inflow(cfd::Parameter &parameter, const cfd::Mesh &mesh,
                                 std::vector<Field> &field, Species &species) {
  // This can also be implemented like the from_start one, which can have patch.
  // But currently, for easy to implement, just initialize the whole flowfield to the main inflow turbulent state.
  // If the need for initialize turbulence in groups is strong,
  // then we implement it just by copying the previous function "initialize_from_start",
  // which should be easy.
  const std::string default_init = parameter.get_string("default_init");
  Inflow inflow(default_init, species, parameter);
  const auto n_turb = parameter.get_int("n_turb");
  const auto n_spec = parameter.get_int("n_spec");
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    const integer mx{mesh[blk].mx}, my{mesh[blk].my}, mz{mesh[blk].mz};
    auto &sv = field[blk].sv;
    for (int k = 0; k < mz; ++k) {
      for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
          for (integer l = 0; l < n_turb; ++l) {
            sv(i, j, k, n_spec + l) = inflow.sv[n_spec + l];
          }
        }
      }
    }
  }
  if (parameter.get_int("myid") == 0) {
    printf("Compute from laminar result. The turbulent field is initialized with freestream.\n");
  }
}

void initialize_mixture_fraction_from_species(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field,
                                              Species &species) {
  // This is called when we need to compute the mixture fraction from a given species field.
  // We need to know the form of coupling functions, the boundary conditions of the 2 streams in order to know how to compute the mixture fraction
  Inflow *fuel = nullptr, *oxidizer = nullptr;

  // First find and initialize the fuel and oxidizer stream
  auto &bcs = parameter.get_string_array("boundary_conditions");
  for (auto &bc_name: bcs) {
    auto &bc = parameter.get_struct(bc_name);
    auto &bc_type = std::get<std::string>(bc.at("type"));
    if (bc_type == "inflow") {
      auto z = std::get<real>(bc.at("mixture_fraction"));
      if (abs(z - 1) < 1e-10) {
        // fuel
        if (fuel == nullptr) {
          fuel = new Inflow(bc_name, species, parameter);
        } else {
          printf("Two fuel stream! Please check the boundary conditions.\n");
        }
      } else if (abs(z) < 1e-10) {
        // oxidizer
        if (oxidizer == nullptr) {
          oxidizer = new Inflow(bc_name, species, parameter);
        } else {
          printf("Two oxidizer stream! Please check the boundary conditions.\n");
        }
      }
    }
  }
  if (fuel == nullptr || oxidizer == nullptr) {
    printf("Cannot find fuel or oxidizer stream! Please check the boundary conditions.\n");
    exit(1);
  }

  // Next, see which definition of mixture fraction is used.
  MixtureFraction *mixtureFraction = nullptr;
  if (species.elem_list.find("C") != species.elem_list.end()) {
    mixtureFraction = new BilgerCH(*fuel, *oxidizer, species, parameter.get_int("myid"));
  } else {
    mixtureFraction = new BilgerH(*fuel, *oxidizer, species, parameter.get_int("myid"));
  }

  std::vector<real> yk(species.n_spec, 0);
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    const integer mx{mesh[blk].mx}, my{mesh[blk].my}, mz{mesh[blk].mz};
    auto &sv = field[blk].sv;
    auto i_fl = parameter.get_int("i_fl");
    for (int k = 0; k < mz; ++k) {
      for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
          for (integer l = 0; l < species.n_spec; ++l) {
            yk[l] = sv(i, j, k, l);
          }
          sv(i, j, k, i_fl) = mixtureFraction->compute_mixture_fraction(yk);
          sv(i, j, k, i_fl + 1) = 0;
        }
      }
    }
  }
  if (parameter.get_int("myid") == 0) {
    printf(
        "Previous results contain only species mass fraction info, the mixture fraction is computed via the Bilger's definition.\n");
  }
}

}

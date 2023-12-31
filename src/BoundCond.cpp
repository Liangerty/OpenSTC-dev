#include "BoundCond.h"
#include "Transport.cuh"
#include "gxl_lib/MyString.h"
#include <cmath>
#include "Parallel.h"

cfd::Inflow::Inflow(const std::string &inflow_name, Species &spec, Parameter &parameter) {
  auto &info = parameter.get_struct(inflow_name);
  label = std::get<integer>(info.at("label"));
  // In default, the mach number, pressure and temperature should be given.
  // If other combinations are given, then implement it later.
  // Currently, 2 combinations are achieved. One is to give (mach, pressure,
  // temperature) The other is to give (density, velocity, pressure)
  if (info.find("mach") != info.end()) mach = std::get<real>(info.at("mach"));
  if (info.find("pressure") != info.end()) pressure = std::get<real>(info.at("pressure"));
  if (info.find("temperature") != info.end()) temperature = std::get<real>(info.at("temperature"));
  if (info.find("velocity") != info.end()) velocity = std::get<real>(info.at("velocity"));
  if (info.find("density") != info.end()) density = std::get<real>(info.at("density"));
  if (info.find("u") != info.end()) u = std::get<real>(info.at("u"));
  if (info.find("v") != info.end()) v = std::get<real>(info.at("v"));
  if (info.find("w") != info.end()) w = std::get<real>(info.at("w"));

  const integer n_scalar = parameter.get_int("n_scalar");
  sv = new real[n_scalar];
  for (int i = 0; i < n_scalar; ++i) {
    sv[i] = 0;
  }
  const integer n_spec{spec.n_spec};

  if (n_spec > 0) {
    // Assign the species mass fraction to the corresponding position.
    // Should be done after knowing the order of species.
    for (auto [name, idx]: spec.spec_list) {
      if (info.find(name) != info.cend()) {
        sv[idx] = std::get<real>(info.at(name));
      }
    }
    mw = 0;
    for (int i = 0; i < n_spec; ++i) mw += sv[i] / spec.mw[i];
    mw = 1 / mw;
  }

  if (temperature < 0) {
    // The temperature is not given, thus the density and pressure are given
    temperature = pressure * mw / (density * R_u);
  }
  if (n_spec > 0) {
    viscosity = compute_viscosity(temperature, mw, sv, spec);
  } else {
    viscosity = Sutherland(temperature);
  }

  real gamma{gamma_air};
  if (n_spec > 0) {
    std::vector<real> cpi(n_spec, 0);
    spec.compute_cp(temperature, cpi.data());
    real cp{0}, cv{0};
    for (size_t i = 0; i < n_spec; ++i) {
      cp += sv[i] * cpi[i];
      cv += sv[i] * (cpi[i] - R_u / spec.mw[i]);
    }
    gamma = cp / cv;  // specific heat ratio
  }

  const real c{std::sqrt(gamma * R_u / mw * temperature)};  // speed of sound

  if (mach < 0) {
    // The mach number is not given. The velocity magnitude should be given
    mach = velocity / c;
  } else {
    // The velocity magnitude is not given. The mach number should be given
    velocity = mach * c;
  }
  u *= velocity;
  v *= velocity;
  w *= velocity;
  if (density < 0) {
    // The density is not given, compute it from equation of state
    density = pressure * mw / (R_u * temperature);
  }
  reynolds_number = density * velocity / viscosity;

  if (parameter.get_int("turbulence_method") == 1) {
    // RANS simulation
    if (parameter.get_int("RANS_model") == 2) {
      // SST
      mut = std::get<real>(info.at("turb_viscosity_ratio")) * viscosity;
      if (info.find("turbulence_intensity") != info.end()) {
        // For SST model, we need k and omega. If SA, we compute this for nothing.
        const real turbulence_intensity = std::get<real>(info.at("turbulence_intensity"));
        sv[n_spec] = 1.5 * velocity * velocity * turbulence_intensity * turbulence_intensity;
        sv[n_spec + 1] = density * sv[n_spec] / mut;
      }
    }
  }

  if ((n_spec > 0 && parameter.get_int("reaction") == 2) || parameter.get_int("species") == 2) {
    // flamelet model or z and z prime are transported
    if (parameter.get_int("turbulence_method") == 1) {
      // RANS simulation
      const auto i_fl{parameter.get_int("i_fl")};
      sv[i_fl] = std::get<real>(info.at("mixture_fraction"));
      sv[i_fl + 1] = 0;
    }
  }

  // This should be re-considered later
  if (inflow_name == parameter.get_string("reference_state")) {
    parameter.update_parameter("rho_inf", density);
    parameter.update_parameter("v_inf", velocity);
    parameter.update_parameter("p_inf", pressure);
    parameter.update_parameter("T_inf", temperature);
    parameter.update_parameter("M_inf", mach);
    parameter.update_parameter("Re_unit", reynolds_number);
    parameter.update_parameter("mu_inf", viscosity);
    std::vector<real> sv_inf(n_scalar, 0);
    for (int l = 0; l < n_scalar; ++l) {
      sv_inf[l] = sv[l];
    }
    parameter.update_parameter("sv_inf", sv_inf);
  }
}

std::tuple<real, real, real, real, real, real> cfd::Inflow::var_info() const {
  return std::make_tuple(density, u, v, w, pressure, temperature);
}

cfd::Wall::Wall(integer type_label, std::ifstream &bc_file) : label(type_label) {
  std::map<std::string, std::string> opt;
  std::map<std::string, double> par;
  std::string input{}, key{}, name{};
  double val{};
  std::istringstream line(input);
  while (gxl::getline_to_stream(bc_file, input, line, gxl::Case::lower)) {
    line >> key;
    if (key == "double") {
      line >> name >> key >> val;
      par.emplace(std::make_pair(name, val));
    } else if (key == "option") {
      line >> name >> key >> key;
      opt.emplace(std::make_pair(name, key));
    }
    if (key == "label" || key == "end") {
      break;
    }
  }
  if (opt.contains("thermal_type")) {
    thermal_type = opt["thermal_type"] == "isothermal" ? ThermalType::isothermal : ThermalType::adiabatic;
  }
  if (thermal_type == ThermalType::isothermal) {
    if (par.contains("temperature")) {
      temperature = par["temperature"];
    } else {
      printf("Isothermal wall does not specify wall temperature, is set as 300K in default.\n");
    }
  }
}

cfd::Wall::Wall(const std::map<std::string, std::variant<std::string, integer, real>> &info)
    : label(std::get<integer>(info.at("label"))) {
  if (info.contains("thermal_type")) {
    thermal_type = std::get<std::string>(info.at("thermal_type")) == "isothermal" ? ThermalType::isothermal
                                                                                  : ThermalType::adiabatic;
  }
  if (thermal_type == ThermalType::isothermal) {
    if (info.contains("temperature")) {
      temperature = std::get<real>(info.at("temperature"));
    } else {
      printf("Isothermal wall does not specify wall temperature, is set as 300K in default.\n");
    }
  }
}

cfd::Symmetry::Symmetry(const std::string &inflow_name, cfd::Parameter &parameter) {
  auto &info = parameter.get_struct(inflow_name);
  label = std::get<integer>(info.at("label"));
}

cfd::Outflow::Outflow(const std::string &inflow_name, cfd::Parameter &parameter) {
  auto &info = parameter.get_struct(inflow_name);
  label = std::get<integer>(info.at("label"));
}

cfd::FarField::FarField(cfd::Species &spec, cfd::Parameter &parameter) {
  auto &info = parameter.get_struct("farfield");
  label = std::get<integer>(info.at("label"));

  // In default, the mach number, pressure and temperature should be given.
  // If other combinations are given, then implement it later.
  // Currently, 3 combinations are achieved.
  // 1. (mach number, pressure, temperature)
  // 2. (density, velocity, pressure)
  // 3. (mach number, temperature, reynolds number)
  if (info.find("mach") != info.end()) mach = std::get<real>(info.at("mach"));
  if (info.find("pressure") != info.end()) pressure = std::get<real>(info.at("pressure"));
  if (info.find("temperature") != info.end()) temperature = std::get<real>(info.at("temperature"));
  if (info.find("velocity") != info.end()) velocity = std::get<real>(info.at("velocity"));
  if (info.find("density") != info.end()) density = std::get<real>(info.at("density"));
  if (info.find("u") != info.end()) u = std::get<real>(info.at("u"));
  if (info.find("v") != info.end()) v = std::get<real>(info.at("v"));
  if (info.find("w") != info.end()) w = std::get<real>(info.at("w"));
  if (info.find("reynolds_number") != info.end()) reynolds_number = std::get<real>(info.at("reynolds_number"));

  const integer n_scalar = parameter.get_int("n_scalar");
  sv = new real[n_scalar];
  for (int i = 0; i < n_scalar; ++i) {
    sv[i] = 0;
  }
  const integer n_spec{spec.n_spec};
  if (n_spec > 0) {
    // Assign the species mass fraction to the corresponding position.
    // Should be done after knowing the order of species.
    for (auto [name, idx]: spec.spec_list) {
      if (info.find(name) != info.cend()) {
        sv[idx] = std::get<real>(info.at(name));
      }
    }
    mw = 0;
    for (int i = 0; i < n_spec; ++i) mw += sv[i] / spec.mw[i];
    mw = 1 / mw;
  }

  if (temperature < 0) {
    // The temperature is not given, thus the density and pressure are given
    temperature = pressure * mw / (density * R_u);
  }
  if (n_spec > 0) {
    viscosity = compute_viscosity(temperature, mw, sv, spec);
  } else {
    viscosity = Sutherland(temperature);
  }

//  real gamma{gamma_air};
  if (n_spec > 0) {
    std::vector<real> cpi(n_spec, 0);
    spec.compute_cp(temperature, cpi.data());
    real cp{0}, cv{0};
    for (size_t i = 0; i < n_spec; ++i) {
      cp += sv[i] * cpi[i];
      cv += sv[i] * (cpi[i] - R_u / spec.mw[i]);
    }
    specific_heat_ratio = cp / cv;  // specific heat ratio
  }

  acoustic_speed = std::sqrt(specific_heat_ratio * R_u / mw * temperature);
//  const real c{std::sqrt(gamma * R_u / mw * temperature)};  // speed of sound

  if (mach < 0) {
    // The mach number is not given. The velocity magnitude should be given
    mach = velocity / acoustic_speed;
  } else {
    // The velocity magnitude is not given. The mach number should be given
    velocity = mach * acoustic_speed;
  }
  u *= velocity;
  v *= velocity;
  w *= velocity;
  if (pressure < 0) {
    // The pressure is not given, which corresponds to case 3, (Ma, temperature, Re)
    density = viscosity * reynolds_number / velocity;
    pressure = density * temperature * R_u / mw;
  }
  if (density < 0) {
    // The density is not given, compute it from equation of state
    density = pressure * mw / (R_u * temperature);
  }
  entropy = pressure / pow(density, specific_heat_ratio);

  if (parameter.get_int("turbulence_method") == 1) {
    // RANS simulation
    if (parameter.get_int("RANS_model") == 2) {
      // SST
      mut = std::get<real>(info.at("turb_viscosity_ratio")) * viscosity;
      if (info.find("turbulence_intensity") != info.end()) {
        // For SST model, we need k and omega. If SA, we compute this for nothing.
        const real turbulence_intensity = std::get<real>(info.at("turbulence_intensity"));
        sv[n_spec] = 1.5 * velocity * velocity * turbulence_intensity * turbulence_intensity;
        sv[n_spec + 1] = density * sv[n_spec] / mut;
      }
    }
  }

  // This should be re-considered later
  if (parameter.get_string("reference_state") == "farfield") {
    parameter.update_parameter("rho_inf", density);
    parameter.update_parameter("v_inf", velocity);
    parameter.update_parameter("p_inf", pressure);
    parameter.update_parameter("T_inf", temperature);
    parameter.update_parameter("M_inf", mach);
    parameter.update_parameter("Re_unit", reynolds_number);
    parameter.update_parameter("mu_inf", viscosity);
    std::vector<real> sv_inf(n_scalar, 0);
    for (int l = 0; l < n_scalar; ++l) {
      sv_inf[l] = sv[l];
    }
    parameter.update_parameter("sv_inf", sv_inf);
  }
}

cfd::SubsonicInflow::SubsonicInflow(const std::string &inflow_name, cfd::Parameter &parameter) {
  const integer n_spec{parameter.get_int("n_spec")};
  if (n_spec > 0) {
    printf("Subsonic inflow boundary condition does not support multi-species simulation.\n");
    MpiParallel::exit();
  }

  auto &info = parameter.get_struct(inflow_name);
  label = std::get<integer>(info.at("label"));

  const real pt_pRef{std::get<real>(info.at("pt_pRef_ratio"))};
  const real Tt_TRef{std::get<real>(info.at("Tt_TRef_ratio"))};
  const real pRef{parameter.get_real("p_inf")};
  const real TRef{parameter.get_real("T_inf")};
  if (info.find("u") != info.end()) u = std::get<real>(info.at("u"));
  if (info.find("v") != info.end()) v = std::get<real>(info.at("v"));
  if (info.find("w") != info.end()) w = std::get<real>(info.at("w"));

  const integer n_scalar = parameter.get_int("n_scalar");
  sv = new real[n_scalar];
  for (int i = 0; i < n_scalar; ++i) {
    sv[i] = 0;
  }

  total_pressure = pt_pRef * pRef;
  total_temperature = Tt_TRef * TRef;

  if (parameter.get_int("turbulence_method") == 1) {
    // RANS simulation
    if (parameter.get_int("RANS_model") == 2) {
      // SST
      const real viscosity{Sutherland(TRef)};
      mut = std::get<real>(info.at("turb_viscosity_ratio")) * viscosity;

      const real velocity{parameter.get_real("v_inf")};
      if (info.find("turbulence_intensity") != info.end()) {
        // For SST model, we need k and omega. If SA, we compute this for nothing.
        const real turbulence_intensity = std::get<real>(info.at("turbulence_intensity"));
        sv[n_spec] = 1.5 * velocity * velocity * turbulence_intensity * turbulence_intensity;
        sv[n_spec + 1] = parameter.get_real("rho_inf") * sv[n_spec] / mut;
      }
    }
  }
}

cfd::BackPressure::BackPressure(const std::string &name, cfd::Parameter &parameter) {
  const integer n_spec{parameter.get_int("n_spec")};
  if (n_spec > 0) {
    printf("Subsonic inflow boundary condition does not support multi-species simulation.\n");
    MpiParallel::exit();
  }

  auto &info = parameter.get_struct(name);
  label = std::get<integer>(info.at("label"));

  if (info.find("pressure") != info.end()) pressure = std::get<real>(info.at("pressure"));
  if (pressure < 0) {
    real p_pRef{1};
    if (info.find("p_pRef_ratio") != info.end()) p_pRef = std::get<real>(info.at("p_pRef_ratio"));
    else {
      printf("Back pressure boundary condition does not specify pressure, is set as 1 in default.\n");
    }
    pressure = p_pRef * parameter.get_real("p_inf");
  }
}

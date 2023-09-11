/**
 * @file BoundCond.h
 * @brief Boundary conditions
 * @details This file contains the definition of boundary conditions. We currently support the following boundary conditions:
 * - 5 Inflow (Inflow) This is the supersonic inflow boundary condition.
 * - 2 Wall (Wall) This is the wall boundary condition.
 * - 6 Outflow (Outflow) This is the supersonic outflow boundary condition.
 * - 3 Symmetry (Symmetry) This is the symmetry boundary condition.
 * - 4 Farfield (FarField) This is the farfield boundary condition with Riemann invariant.
 * - 7 Subsonic inflow (SubsonicInflow) This is the subsonic inflow boundary condition.
 * @version 0.1
 * @date 2023-09-08
 * @note Initial version of the file with basic boundary conditions.
 *
 */
#pragma once

#include "ChemData.h"
#include "Constants.h"

namespace cfd {

struct Inflow {
  explicit Inflow(const std::string &inflow_name, Species &spec, Parameter &parameter);

  [[nodiscard]] std::tuple<real, real, real, real, real, real> var_info() const;

  void copy_to_gpu(Inflow *d_inflow, Species &spec, const Parameter &parameter);

  integer label = 5;
  real mach = -1;
  real pressure = 101325;
  real temperature = -1;
  real velocity = 0;
  real density = -1;
  real u = 1, v = 0, w = 0;
  real *sv = nullptr;
  real mw = mw_air;
  real viscosity = 0;
  real mut = 0;
  real reynolds_number = -1;
};

struct Wall {
  explicit Wall(integer type_label, std::ifstream &bc_file);

  explicit Wall(const std::map<std::string, std::variant<std::string, integer, real>> &info);

  enum class ThermalType {
    isothermal, adiabatic
  };

  integer label = 2;
  ThermalType thermal_type = ThermalType::isothermal;
  real temperature{300};
};

struct Outflow {
  explicit Outflow(const std::string &inflow_name, cfd::Parameter &parameter);

  integer label = 6;
};

struct Symmetry {
  explicit Symmetry(const std::string &inflow_name, cfd::Parameter &parameter);

  integer label = 3;
};

struct FarField {
  explicit FarField(cfd::Species &spec, cfd::Parameter &parameter);

  void copy_to_gpu(FarField *d_farfield, Species &spec, const Parameter &parameter);

  integer label = 4;
  real mach = -1;
  real pressure = -1;
  real temperature = -1;
  real velocity = 0;
  real density = -1;
  real u = 1, v = 0, w = 0;
  real acoustic_speed = 0;
  real entropy = 0;
  real *sv = nullptr;
  real mw = mw_air;
  real viscosity = 0;
  real mut = 0;
  real specific_heat_ratio = gamma_air;
  real reynolds_number = -1;
};

/**
 * @brief Subsonic inflow boundary condition, default label is 7
 * @details This is the subsonic inflow boundary condition.
 * @note We need to specify the ratio of total pressure and reference pressure, the ratio of total temperature and reference temperature, and the velocity direction in u, v, and w.
 * The implementation references the Fun3D manual and the CFL3D code.
 * @refitem Carlson, J.-R. Inflow/Outflow Boundary Conditions with Application to FUN3D. (2011).
 * @warning This implementation can only be used in air simulation. The multi-species simulation is not supported.
 */
struct SubsonicInflow {
  explicit SubsonicInflow(const std::string &inflow_name, cfd::Parameter &parameter);

  void copy_to_gpu(SubsonicInflow *d_inflow, Species &spec, const Parameter &parameter);

  integer label = 7;
//  real mach = -1;
  real total_pressure = 101325;
  real total_temperature = -1;
//  real pressure = 101325;
//  real temperature = -1;
//  real velocity = 0;
//  real density = -1;
  real u = 1, v = 0, w = 0;
  real *sv = nullptr;
//  real mw = mw_air;
//  real viscosity = 0;
  real mut = 0;
};

/**
 * @brief Back pressure boundary condition, default label is 9
 * @details This boundary condition specifies the back pressure at the outlet.
 * @note This boundary condition is only used in the subsonic flow simulation, which you make sure that the outflow is subsonic.
 * All quantities except pressure is extrapolated from the interior, while the pressure is specified by the user as the back pressure.
 */
struct BackPressure {
  explicit BackPressure(const std::string &name, cfd::Parameter &parameter);

  integer label = 9;
  real pressure = -1;
};

}

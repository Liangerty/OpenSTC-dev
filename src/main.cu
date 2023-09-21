#include <cstdio>
#include "Parallel.h"
#include "Parameter.h"
#include "Mesh.h"
#include "Driver.cuh"
#include "Simulate.cuh"

int main(int argc, char *argv[]) {
  cfd::MpiParallel mpi_parallel(&argc, &argv);

  cfd::Parameter parameter(mpi_parallel);

  cfd::Mesh mesh(parameter);

  bool species = parameter.get_bool("species");
  bool turbulent_laminar = parameter.get_bool("turbulence");
  integer reaction = parameter.get_int("reaction");
  integer turbulent_method = parameter.get_int("turbulence_method");
  if (!turbulent_laminar) {
    turbulent_method = 0;
  }

  if (species) {
    // Multiple species
    if (turbulent_method == 1) {
      // RANS
      if (reaction == 1) {
        // Finite rate chemistry
        cfd::Driver<MixtureModel::FR, TurbulenceMethod::RANS> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
//        driver.simulate();
      } else if (reaction == 2) {
        // Flamelet model
        cfd::Driver<MixtureModel::FL, TurbulenceMethod::RANS> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      } else {
        // Pure mixing among species
        cfd::Driver<MixtureModel::Mixture, TurbulenceMethod::RANS> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      }
    } else {
      // Laminar
      if (reaction == 1) {
        // Finite rate chemistry
        cfd::Driver<MixtureModel::FR, TurbulenceMethod::Laminar> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      } else {
        // Pure mixing among species
        cfd::Driver<MixtureModel::Mixture, TurbulenceMethod::Laminar> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      }
    }
  } else {
    // Air simulation
    if (turbulent_method == 1) {
      // RANS and air
      cfd::Driver<MixtureModel::Air, TurbulenceMethod::RANS> driver(parameter, mesh);
      driver.initialize_computation();
      simulate(driver);
    } else {
      // Laminar and air
      cfd::Driver<MixtureModel::Air, TurbulenceMethod::Laminar> driver(parameter, mesh);
      driver.initialize_computation();
      simulate(driver);
    }
  }

  printf("Fuck off\n");
  return 0;
}

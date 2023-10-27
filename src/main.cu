#include <cstdio>
#include "Parallel.h"
#include "Parameter.h"
#include "Mesh.h"
#include "Driver.cuh"
#include "Simulate.cuh"
#include "SST.cuh"

int main(int argc, char *argv[]) {
  cfd::MpiParallel mpi_parallel(&argc, &argv);

  cfd::Parameter parameter(mpi_parallel);

  cfd::Mesh mesh(parameter);

  integer species = parameter.get_int("species");
  bool turbulent_laminar = parameter.get_bool("turbulence");
  integer reaction = parameter.get_int("reaction");
  integer turbulent_method = parameter.get_int("turbulence_method");
  if (!turbulent_laminar) {
    parameter.update_parameter("turbulence_method", 0);
    turbulent_method = 0;
  }

  if (species == 1) {
    // Multiple species
    if (turbulent_method == 1) {
      // RANS
      if (reaction == 1) {
        // Finite rate chemistry
        cfd::Driver<MixtureModel::FR, cfd::SST> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      } else if (reaction == 2) {
        // Flamelet model
        cfd::Driver<MixtureModel::FL, cfd::SST> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      } else {
        // Pure mixing among species
        cfd::Driver<MixtureModel::Mixture, cfd::SST> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      }
    } else {
      // Laminar
      if (reaction == 1) {
        // Finite rate chemistry
        cfd::Driver<MixtureModel::FR, cfd::Laminar> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      } else {
        // Pure mixing among species
        cfd::Driver<MixtureModel::Mixture, cfd::Laminar> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      }
    }
  } else if (species == 2) {
    // Mixture fraction and mixture fraction variance are solved together with species mixing.
    if (turbulent_method == 1) {
      // RANS
      if (reaction == 0) {
        // Compute the species mixing, with mixture fraction and mixture fraction variance also solved.
        cfd::Driver<MixtureModel::MixtureFraction, cfd::SST> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      } else if (reaction == 2) {
        // Flamelet model
        cfd::Driver<MixtureModel::FL, cfd::SST> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      } else {
        printf("The combination of species model 2 and reaction model %d is not implemented", reaction);
      }
    } else {
      // Laminar
      cfd::Driver<MixtureModel::MixtureFraction, cfd::Laminar> driver(parameter, mesh);
      driver.initialize_computation();
      simulate(driver);
    }

  } else {
    // Air simulation
    if (turbulent_method == 1) {
      // RANS and air
      cfd::Driver<MixtureModel::Air, cfd::SST> driver(parameter, mesh);
      driver.initialize_computation();
      simulate(driver);
    } else {
      // Laminar and air
      cfd::Driver<MixtureModel::Air, cfd::Laminar> driver(parameter, mesh);
      driver.initialize_computation();
      simulate(driver);
    }
  }

  printf("Fuck off\n");
  return 0;
}

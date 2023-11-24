#include "Parameter.h"
#include <fstream>
#include <sstream>
#include "Parallel.h"
#include <filesystem>
#include "gxl_lib/MyString.h"

cfd::Parameter::Parameter(const MpiParallel &mpi_parallel) {
  read_param_from_file();
  int_parameters["myid"] = mpi_parallel.my_id;
  int_parameters["n_proc"] = mpi_parallel.n_proc;
  bool_parameters["parallel"] = cfd::MpiParallel::parallel;
}

cfd::Parameter::Parameter(const std::string &filename) {
  std::ifstream file(filename);
  read_one_file(file);
  file.close();
}

void cfd::Parameter::read_param_from_file() {
  for (auto &name: file_names) {
    std::ifstream file(name);
    read_one_file(file);
    file.close();
  }

  int_parameters.emplace("step", 0);

  deduce_known_info();

  update_parameter("n_var", 5);
  update_parameter("n_turb", 0);
  integer n_scalar{0};
  if (bool_parameters["turbulence"] == 1) {
    if (int_parameters["turbulence_method"] == 1) { //RANS
      if (int_parameters["RANS_model"] == 1) {// SA
        update_parameter("n_turb", 1);
        update_parameter("n_var", 5 + 1);
        n_scalar += 1;
      } else { // SST
        update_parameter("n_turb", 2);
        update_parameter("n_var", 5 + 2);
        n_scalar += 2;
      }
    }
  } else {
    update_parameter("RANS_model", 0);
  }
  update_parameter("n_scalar", n_scalar);
}

void cfd::Parameter::read_one_file(std::ifstream &file) {
  std::string input{}, type{}, key{}, temp{};
  std::istringstream line(input);
  while (std::getline(file, input)) {
    if (input.starts_with("//") || input.starts_with("!") || input.empty()) {
      continue;
    }
    line.clear();
    line.str(input);
    line >> type;
    line >> key >> temp;
    if (type == "int") {
      int val{};
      line >> val;
      int_parameters[key] = val;
    } else if (type == "real") {
      real val{};
      line >> val;
      real_parameters[key] = val;
    } else if (type == "bool") {
      bool val{};
      line >> val;
      bool_parameters[key] = val;
    } else if (type == "string") {
      std::string val{};
      line >> val;
      string_parameters[key] = val;
    } else if (type == "array") {
      if (key == "int") {
        std::vector<int> arr;
        std::string name{temp};
        line >> temp; // {
        while (read_line_to_array(line, arr)) {
          gxl::getline_to_stream(file, input, line);
        }
        int_array[name] = arr;
      } else if (key == "real") {
        std::vector<real> arr;
        std::string name{temp};
        line >> temp; // {
        while (read_line_to_array(line, arr)) {
          gxl::getline_to_stream(file, input, line);
        }
        real_array[name] = arr;
      } else if (key == "string") {
        std::vector<std::string> arr;
        std::string name{temp};
        line >> temp >> temp;
        while (read_line_to_array(line, arr)) {
          gxl::getline_to_stream(file, input, line);
        }
        string_array[name] = arr;
      }
    } else if (type == "struct") {
      auto the_struct = read_struct(file);
      struct_array[key] = the_struct;
    }
  }
  file.close();
}

template<typename T>
integer cfd::Parameter::read_line_to_array(std::istringstream &line, std::vector<T> &arr) {
  std::string temp{};
  while (line >> temp) {
    if (temp == "}") {
      // Which means the array has been read
      return 0;
    }
    if (temp == "//") {
      // which means the array is not over, but values are on the next line
      break;
    }
    T val{};
    if constexpr (std::is_same_v<T, real>) {
      val = std::stod(temp);
    } else if constexpr (std::is_same_v<T, integer>) {
      val = std::stoi(temp);
    } else if constexpr (std::is_same_v<T, std::string>) {
      val = temp;
    }
    arr.push_back(val);
  }
  return 1; // Which means we need to read the next line
}

std::map<std::string, std::variant<std::string, integer, real>> cfd::Parameter::read_struct(std::ifstream &file) {
  std::map<std::string, std::variant<std::string, integer, real>> struct_to_read;
  std::string input{}, key{}, temp{};
  std::istringstream line(input);
  while (gxl::getline_to_stream(file, input, line)) {
    line >> key;
    if (key == "}") { // the "}" must be placed on a separate line.
      break;
    }
    if (key == "string") {
      std::string val{};
      line >> key >> temp >> val;
      struct_to_read.emplace(std::make_pair(key, val));
    } else if (key == "int") {
      integer val{};
      line >> key >> temp >> val;
      struct_to_read.emplace(std::make_pair(key, val));
    } else if (key == "real") {
      real val{};
      line >> key >> temp >> val;
      struct_to_read.emplace(std::make_pair(key, val));
    }
  }
  return struct_to_read;
}

void cfd::Parameter::deduce_known_info() {
  // Ghost grid number is decided from the inviscid scheme and viscous scheme.
  // For common 2nd order schemes, we need 2 ghost grids.
  // Here, because we now only implement 2nd order central difference for viscous flux, we need 2 ghost grids.
  // The main concern for the number of ghost grids is the inviscid flux, which in turn is decided from the reconstruction method.
  integer ngg{2};
  integer reconstruction_scheme = get_int("reconstruction");
  switch (reconstruction_scheme) {
    case 4: // WENO5
      ngg = 3;
      break;
    case 5: // WENO7
      ngg = 4;
      break;
    default: // 1st, MUSCL, NND2, etc.
      break;
  }
  update_parameter("ngg", ngg);

  // Next, based on the reconstruction scheme and inviscid flux method, we re-assign a new field called "inviscid_tag"
  // to identify the method to be used.
  // 2-Roe, 3-AUSM+, 4-HLLC. These are used by default if the corresponding reconstruction methods are 1stOrder, MUSCL, or NND2.
  // Once the reconstruction method is changed to high-order ones, such as WENO5, WENO7, etc., we would use a new tag to identify
  // them. Because, the reconstructed variables would be conservative variables, and there may be additional operations,
  // such as characteristic reconstruction, high-order central difference, etc.
  // Summary: 2-Roe, 3-AUSM+, 4-HLLC, 14-HLLC+WENO
  integer inviscid_tag{get_int("inviscid_scheme")};
  integer inviscid_type{0};
  if (inviscid_tag == 2) {
    inviscid_type = 2;
  }
  if (reconstruction_scheme == 4 || reconstruction_scheme == 5) {
    inviscid_type = 1;
    // WENO reconstructions
    switch (inviscid_tag) {
      case 1: // LF + WENO
        inviscid_tag = 11;
        break;
      case 2: // Roe + WENO, which is not supported yet.
        printf("Roe + WENO is not supported yet.\n");
        MpiParallel::exit();
      case 3: // AUSM + WENO, which is not supported yet.
        printf("AUSM + WENO is not supported yet.\n");
        MpiParallel::exit();
      case 4: // HLLC + WENO, currently assigned by tag 14.
        inviscid_tag = 14;
        break;
      default:
        printf("??? + WENO is not supported yet.\n");
        MpiParallel::exit();
    }
  }
  update_parameter("inviscid_tag", inviscid_tag);
  update_parameter("inviscid_type", inviscid_type);
}

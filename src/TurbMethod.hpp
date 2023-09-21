#pragma once

namespace cfd {
enum class TurbulentSimulationMethod{
  Lam,
  RANS,
  DES,
  LES,
  DNS
};

template<typename T>
struct TurbMethod {
  static constexpr bool isLaminar = true;
  static constexpr bool hasMut = false;
  static constexpr TurbulentSimulationMethod type = TurbulentSimulationMethod::Lam;
};

struct Laminar{};


}
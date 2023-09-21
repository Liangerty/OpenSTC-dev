#pragma once

namespace cfd {
template<typename T>
struct TurbMethod {
  static constexpr bool isLaminar = true;
  static constexpr bool hasMut = false;
};

struct Laminar{};


}
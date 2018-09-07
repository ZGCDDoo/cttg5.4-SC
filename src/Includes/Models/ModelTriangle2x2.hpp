#pragma once

#include "ABC_Model.hpp"
#include "H0Triangle.hpp"

namespace Models
{

class ModelTriangle2x2 : public ABC_Model_2D<IO::IOTriangle2x2, H0Triangle<Nx2, Nx2>>
{

  using IOModel_t = IO::IOTriangle2x2;
  using H0_t = H0Triangle<Nx2, Nx2>;

public:
  ModelTriangle2x2(const Json &jj) : ABC_Model_2D<IOModel_t, H0_t>(jj){};
};
} // namespace Models
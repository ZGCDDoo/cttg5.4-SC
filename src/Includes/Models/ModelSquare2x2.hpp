#pragma once

#include "ABC_Model.hpp"
#include "H0Square.hpp"

namespace Models
{

class ModelSquare2x2 : public ABC_Model_2D<IO::IOSquare2x2, H0Square<Nx2, Nx2>>
{

  using IOModel_t = IO::IOSquare2x2;
  using H0_t = H0Square<Nx2, Nx2>;

public:
  ModelSquare2x2(const Json &jj) : ABC_Model_2D<IOModel_t, H0_t>(jj){};
};
} // namespace Models
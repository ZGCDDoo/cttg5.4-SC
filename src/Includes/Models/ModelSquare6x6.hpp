#pragma once

#include "ABC_Model.hpp"
#include "H0Square.hpp"

namespace Models
{

class ModelSquare6x6 : public ABC_Model_2D<IO::IOSquare6x6, H0Square<Nx6, Nx6>>
{

  using H0_t = H0Square<Nx6, Nx6>;
  using IOModel_t = IO::IOSquare6x6;

public:
 
  ModelSquare6x6(const Json &jj) : ABC_Model_2D<IOModel_t, H0_t>(jj){};
};
} // namespace Models
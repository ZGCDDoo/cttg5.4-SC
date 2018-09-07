#pragma once

#include "ABC_Model.hpp"
#include "H0Square.hpp"

namespace Models
{

class ModelSquare4x4 : public ABC_Model_2D<IO::IOSquare4x4, H0Square<Nx4, Nx4>>
{

  using H0_t = H0Square<Nx4, Nx4>;
  using IOModel_t = IO::IOSquare4x4;

public:
  
  ModelSquare4x4(const Json &jj) : ABC_Model_2D<IOModel_t, H0_t>(jj){};
};
} // namespace Models
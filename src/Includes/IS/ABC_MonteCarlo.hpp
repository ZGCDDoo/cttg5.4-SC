#pragma once

#include "../Utilities/Utilities.hpp"

namespace MC
{

class ABC_MonteCarlo
{
public:
  ABC_MonteCarlo(){};

  virtual ~ABC_MonteCarlo() = 0;

  virtual void RunMonteCarlo() = 0;
}; // class ABC_MonteCarlo

ABC_MonteCarlo::~ABC_MonteCarlo() {} //destructors must exist

} // namespace MC

#pragma once

#include "Utilities.hpp"

namespace SelfCon
{

class ABC_SelfConsistency
{
public:
  ABC_SelfConsistency(){};
  virtual ~ABC_SelfConsistency() = 0;
  virtual void DoSCGrid() = 0;

private:
}; // class ABC_SelfConsistency

ABC_SelfConsistency::~ABC_SelfConsistency() {} //destructors must exist
} // namespace SelfCon
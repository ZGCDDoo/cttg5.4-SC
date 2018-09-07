#pragma once

#include "ABC_MarkovChain.hpp"

namespace Markov
{

template <typename TIOModel, typename TModel>
class MarkovChain : public ABC_MarkovChain<TIOModel, TModel>
{

public:
  MarkovChain(const Json &jj, const size_t &seed) : ABC_MarkovChain<TIOModel, TModel>(jj, seed){};

  ~MarkovChain(){};

  //Overriding
  double gammaUpTrad(const AuxSpin_t &auxTo, const AuxSpin_t &auxFrom) override { return (this->modelPtr_->gammaUp(auxTo, auxFrom)); }
  double gammaDownTrad(const AuxSpin_t &auxTo, const AuxSpin_t &auxFrom) override { return (this->modelPtr_->gammaDown(auxTo, auxFrom)); }
  double KAux() override { return (this->modelPtr_->KAux()); }
  double FAuxUp(const AuxSpin_t &aux) override { return (this->modelPtr_->FAuxUp(aux)); }
  double FAuxDown(const AuxSpin_t &aux) override { return (this->modelPtr_->FAuxDown(aux)); }

private:
};
} // namespace Markov

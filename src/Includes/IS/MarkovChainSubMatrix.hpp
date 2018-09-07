#pragma once

#include "ABC_MarkovChainSubMatrix.hpp"

namespace Markov
{

template <typename TIOModel, typename TModel>
class MarkovChainSub : public ABC_MarkovChainSubMatrix<TIOModel, TModel>
{

  public:
    MarkovChainSub(const Json &jj, const size_t &seed) : ABC_MarkovChainSubMatrix<TIOModel, TModel>(jj, seed){};

    ~MarkovChainSub(){};

    //Overriding
    double gammaUpSubMatrix(const AuxSpin_t &auxTo, const AuxSpin_t &auxFrom) override { return (this->model_.gammaUp(auxTo, auxFrom)); }
    double gammaDownSubMatrix(const AuxSpin_t &auxTo, const AuxSpin_t &auxFrom) override { return (this->model_.gammaDown(auxTo, auxFrom)); }
    double KAux() override { return (this->model_.KAux()); }
    double FAuxUp(const AuxSpin_t &aux) override { return (this->model_.FAuxUp(aux)); }
    double FAuxDown(const AuxSpin_t &aux) override { return (this->model_.FAuxDown(aux)); }

  private:
};
}
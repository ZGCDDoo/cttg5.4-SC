#pragma once

#include "ABC_MarkovChain.hpp"

namespace Markov
{

template <typename TIOModel, typename TModel>
class MarkovChainAux : public ABC_MarkovChain<TIOModel, TModel>
{

  public:
    MarkovChainAux(const Json &jj, const size_t &seed) : ABC_MarkovChain<TIOModel, TModel>(jj, seed){};

    ~MarkovChainAux(){};

    //Overriding
    double gammaUpTrad(const AuxSpin_t &auxTo, const AuxSpin_t &auxFrom) override
    {
        double fsJ = FAuxUp(auxFrom);
        return ((FAuxUp(auxTo) - fsJ) / fsJ);
    }

    double gammaDownTrad(const AuxSpin_t &auxTo, const AuxSpin_t &auxFrom) override
    {
        double fsJ = FAuxDown(auxFrom);
        return ((FAuxDown(auxTo) - fsJ) / fsJ);
    }

    double KAux() override { return (this->modelPtr_->K()); }

    double FAuxUp(const AuxSpin_t &aux) override
    {
        if (aux == AuxSpin_t::Zero)
        {
            return 1.0;
        }
        return (aux == AuxSpin_t::Up ? this->expUp_ : this->expDown_);
    }

    double FAuxDown(const AuxSpin_t &aux) override
    {
        if (aux == AuxSpin_t::Zero)
        {
            return 1.0;
        }
        return (aux == AuxSpin_t::Down ? this->expUp_ : this->expDown_);
    }

  private:
};
}
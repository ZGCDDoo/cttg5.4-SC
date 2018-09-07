#pragma once

#ifdef DCA
#include "ABC_H0_DCA.hpp"
#else
#include "ABC_H0.hpp"
#endif

namespace Models
{
template <size_t TNX, size_t TNY>
class H0Square : public ABC_H0<TNX, TNY>
{

  public:
    //H0Square(){};
    H0Square(const double &t, const double &tp, const double &tpp) : ABC_H0<TNX, TNY>(t, tp, tpp) {}

    double Eps0k(const double &kx, const double &ky) const
    {
        return (2.0 * (this->t_ * (std::cos(kx) + std::cos(ky))) + 2.0 * this->tPrime_ * (std::cos(kx + ky) + std::cos(kx - ky)) + 2.0 * this->tPrimePrime_ * (std::cos(2.0 * kx) + std::cos(2.0 * ky)));
    }
};
} // namespace Models
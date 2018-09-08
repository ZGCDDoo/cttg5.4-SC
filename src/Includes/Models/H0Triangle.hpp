#pragma once


#include "ABC_H0.hpp"


namespace Models
{
template <size_t TNX, size_t TNY>
class H0Triangle : public ABC_H0<TNX, TNY>
{

  public:
    H0Triangle(const double &t, const double &tp, const double &tpp) : ABC_H0<TNX, TNY>(t, tp, tpp) {}

    double Eps0k(const double &kx, const double &ky) const
    {
        return (
            2.0 * (this->t_ * (std::cos(kx) + std::cos(ky)) +
                   this->tPrime_ * (std::cos(kx + ky))));
    }
};
} // namespace Models
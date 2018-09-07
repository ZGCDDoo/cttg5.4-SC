#pragma once
#include "../Utilities/Utilities.hpp"
#include "../Utilities/Fourier_DCA.hpp"
#include "../Utilities/Integrator.hpp"

namespace Models
{
template <size_t TNX, size_t TNY>
class ABC_H0
{

  public:
    static const size_t Nc;
    static const size_t Nx;
    static const size_t Ny;
    static const size_t n_rows;
    static const size_t n_cols;
    const size_t NKPTS = 1000;

    ABC_H0(const double &t, const double &tp, const double &tpp) : RSites_(Nc),
                                                                   KWaveVectors_(Nc),
                                                                   t_(t),
                                                                   tPrime_(tp),
                                                                   tPrimePrime_(tpp)

    {
        assert(TNX == TNY);

        for (size_t i = 0; i < TNX; i++)
        {
            for (size_t j = 0; j < TNY; j++)
            {
                const size_t index = i + TNY * j;
                RSites_.at(index) = {static_cast<double>(i), static_cast<double>(j)};
                KWaveVectors_.at(index) = {static_cast<double>(i) * 2.0 * M_PI / static_cast<double>(TNX), static_cast<double>(j) * 2.0 * M_PI / static_cast<double>(TNY)};
            }
        }

        assert(KWaveVectors_.size() == Nc);
        assert(KWaveVectors_.size() == RSites_.size());
    }

    virtual ~ABC_H0() = 0;

    double t() const { return t_; };
    double tPrime() const { return tPrime_; };
    double tPrimePrime() const { return tPrimePrime_; };
    ClusterSites_t RSites() const { return RSites_; };
    ClusterSites_t KWaveVectors() const { return KWaveVectors_; };

    virtual double Eps0k(const double &kx, const double &ky) const = 0;

    ClusterMatrixCD_t operator()(const double &kTildeX, const double &kTildeY) //return t(ktilde)
    {
        const cd_t im = cd_t(0.0, 1.0);
        const SiteVector_t ktilde = {kTildeX, kTildeY};
        ClusterMatrixCD_t HoppingKTilde(Nc, Nc);
        HoppingKTilde.zeros();

        for (size_t i = 0; i < Nc; i++)
        {
            for (size_t j = 0; j < Nc; j++)
            {
                for (const SiteVector_t &K : this->KWaveVectors_)
                {
                    HoppingKTilde(i, j) += std::exp(im * dot(K, RSites_.at(i) - RSites_[j])) * Eps0k(K(0) + kTildeX, K(1) + kTildeY);
                }
            }
        }
        return (HoppingKTilde / static_cast<double>(Nc));
    }

  
  protected:
    ClusterSites_t RSites_;
    ClusterSites_t KWaveVectors_;

    const double t_;
    const double tPrime_;
    const double tPrimePrime_;
};

template <size_t TNX, size_t TNY>
ABC_H0<TNX, TNY>::~ABC_H0() {} //destructors must exist

template <size_t TNX, size_t TNY>
const size_t ABC_H0<TNX, TNY>::Nx = TNX;

template <size_t TNX, size_t TNY>
const size_t ABC_H0<TNX, TNY>::Ny = TNY;

template <size_t TNX, size_t TNY>
const size_t ABC_H0<TNX, TNY>::Nc = TNX *TNY;

template <size_t TNX, size_t TNY>
const size_t ABC_H0<TNX, TNY>::n_rows = TNX *TNY;

template <size_t TNX, size_t TNY>
const size_t ABC_H0<TNX, TNY>::n_cols = TNX *TNY;
} // namespace Models
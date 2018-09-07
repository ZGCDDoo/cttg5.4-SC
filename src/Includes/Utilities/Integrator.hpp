
#pragma once

#include "../../deps/cubature/cubature.h"
#include "Utilities.hpp"

namespace Integrator
{
const size_t MAXEVALS = 100000;
const size_t KXPTS = 64;

template <typename TFct>
ClusterMatrixCD_t GridKTilde(TFct fct, size_t kxpts = KXPTS)
{
    double fact = 1.0 / (static_cast<double>(kxpts * kxpts));
    ClusterMatrixCD_t integral(fct.Nc, fct.Nc);
    integral.zeros();

    for (size_t ii = 0; ii < kxpts; ii++)
    {
        double kx = static_cast<double>(ii) / static_cast<double>(kxpts) * 2.0 * M_PI / static_cast<double>(fct.Nx);
        for (size_t jj = 0; jj < kxpts; jj++)
        {
            const double ky = static_cast<double>(jj) / static_cast<double>(kxpts) * 2.0 * M_PI / static_cast<double>(fct.Ny);
            integral += fct(kx, ky);
        }
    }
    return (fact * integral);
}
template <typename TFct>
int IntegrandCubature(unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval)
{
    // TFct est un struct ou une classe qui implémente les éléments suivants:
    // TFct fct: fct(x, y), appelle a la fct, fct.n_cols, fct.n_rows, la dimensionalité est de 2
    size_t tmp_to_shutup_warning = fdim + ndim; //to shutup unused variable.
    tmp_to_shutup_warning++;

    TFct fct = *((TFct *)fdata);

    ClusterMatrixCD_t tmpMat;
    for (size_t i = 0; i < fct.n_rows; i++)
    {
        for (size_t j = 0; j < fct.n_cols; j++)
        {
            //std::cout << " x[0] = " << x[0]
            //        << std::endl;
            tmpMat = fct(x[0], x[1]);
            fval[i + fct.n_rows * j] = tmpMat(i, j).real();
            fval[i + fct.n_rows * (fct.n_cols + j)] = tmpMat(i, j).imag();
        }
    }

    return 0; // success
}

template <typename TFct>
ClusterMatrixCD_t Cubature(TFct fct, double *xmin, double *xmax, size_t maxevals = MAXEVALS, double absError = 1.49e-6, double relError = 1.49e-6)
{
    unsigned nelem = fct.n_rows * fct.n_cols;
    double *val = new double[2 * nelem]; //for complex values
    double *err = new double[2 * nelem];

    //std::cout << "before calling hcubature in integrator.hpp " << 2 * nelem << std::endl;

    hcubature(2 * nelem, IntegrandCubature<TFct>, &fct, 2, xmin, xmax, maxevals, absError, relError, ERROR_INDIVIDUAL, val, err);

    ClusterMatrixCD_t result(fct.n_rows, fct.n_cols);

    for (size_t i = 0; i < fct.n_rows; i++)
    {
        for (size_t j = 0; j < fct.n_cols; j++)
        {
            double real = val[i + fct.n_rows * j];
            double imag = val[i + fct.n_rows * (fct.n_cols + j)];
            result(i, j) = cd_t(real, imag);
        }
    }

    return result;
}

template <typename TFct>
ClusterMatrixCD_t CubatureKTilde(TFct fct, size_t maxevals = MAXEVALS)
{
    double xmin[2] = {-M_PI / (fct.Nx), -M_PI / (fct.Ny)};
    double xmax[2] = {M_PI / (fct.Nx), M_PI / (fct.Ny)};

    double fact = fct.Nc / (4.0 * M_PI * M_PI);
    return (fact * Cubature(fct, xmin, xmax, maxevals));
}

} // namespace Integrator
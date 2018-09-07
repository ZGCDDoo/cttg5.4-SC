#pragma once

#include "Utilities.hpp"
#include "MPIUtilities.hpp"
namespace Cache
{

using cachearray_t = std::vector<std::vector<cd_t>>;

cachearray_t ComputeExps(double beta, size_t NMat, size_t NTau)
{
    cd_t iwn;
    double tau;
    cachearray_t result(NMat);

    mpiUt::Print("In ComputeExps ");
    for (size_t nn = 0; nn < NMat; nn++)
    {
        iwn = cd_t(0.0, (2 * nn + 1.0) * M_PI / beta);
        result.at(nn).resize(2 * NTau + 1);
        for (size_t ii = 0; ii < 2 * NTau + 1; ii++)
        {
            tau = -beta + static_cast<double>(ii) / static_cast<double>(NTau) * beta;
            result.at(nn).at(ii) = std::exp(iwn * tau);
        }
    }

    //you than access the desired exponential for a given iwn and a given tau by the following:
    //size_t tau_index = static_cast<size_t>((tau + beta) / beta * static_cast<double>(NTau));
    //exp(iwn*tau) = result.at(nn).at(tau_index)

    mpiUt::Print("After ComputeExps ");
    return result;
}
}
#pragma once

#include "Integrator.hpp"
#include "Utilities.hpp"
#include "MPIUtilities.hpp"
#include "ABC_SelfConsistency.hpp"

namespace FourierDCA
{
using DataK_t = ClusterCubeCD_t; //first index (row) = K, second index (col) =iwn

DataK_t RtoK(const ClusterCubeCD_t &greenR, const ClusterSites_t &RSites, const ClusterSites_t &KWaveVectors)
{
    assert(RSites.size() == KWaveVectors.size());
    DataK_t greenK(KWaveVectors.size(), KWaveVectors.size(), greenR.n_slices);
    greenK.zeros();

    //for each matsubara freq.
    const cd_t im = cd_t(0.0, 1.0);
    for (size_t nn = 0; nn < greenR.n_slices; nn++)
    {

        for (size_t KIndex1 = 0; KIndex1 < KWaveVectors.size(); KIndex1++)
        {

            for (size_t RIndex1 = 0; RIndex1 < RSites.size(); RIndex1++)
            {
                for (size_t RIndex2 = 0; RIndex2 < RSites.size(); RIndex2++)
                {

                    greenK(KIndex1, KIndex1, nn) += std::exp(-im * dot(KWaveVectors.at(KIndex1), RSites.at(RIndex1))) * std::exp(im * dot(KWaveVectors.at(KIndex1), RSites.at(RIndex2))) * greenR(RIndex1, RIndex2, nn);
                }
            }
        }
    }
    return (1.0 / static_cast<double>(RSites.size()) * greenK);
}

ClusterCubeCD_t KtoR(const DataK_t &greenK, const ClusterSites_t &RSites, const ClusterSites_t &KWaveVectors)
{
    assert(RSites.size() == KWaveVectors.size());
    ClusterCubeCD_t greenR(KWaveVectors.size(), RSites.size(), greenK.n_slices);
    greenR.zeros();

    //for each matsubara freq.
    const cd_t im = cd_t(0.0, 1.0);
    for (size_t nn = 0; nn < greenR.n_slices; nn++)
    {

        for (size_t RIndex1 = 0; RIndex1 < RSites.size(); RIndex1++)
        {
            for (size_t RIndex2 = 0; RIndex2 < RSites.size(); RIndex2++)
            {
                for (size_t KIndex1 = 0; KIndex1 < KWaveVectors.size(); KIndex1++)
                {

                    greenR(RIndex1, RIndex2, nn) += std::exp(im * dot(KWaveVectors.at(KIndex1), RSites.at(RIndex1))) * std::exp(-im * dot(KWaveVectors.at(KIndex1), RSites.at(RIndex2))) * greenK(KIndex1, KIndex1, nn);
                }
            }
        }
    }
    return (1.0 / static_cast<double>(RSites.size()) * greenR);
}

ClusterMatrixCD_t RtoK(const ClusterMatrixCD_t &greenR, const ClusterSites_t &RSites, const ClusterSites_t &KWaveVectors)
{
    const size_t Nc = KWaveVectors.size();
    assert(RSites.size() == Nc);
    ClusterMatrixCD_t greenK(Nc, Nc);
    greenK.zeros();

    const cd_t im = cd_t(0.0, 1.0);
    for (size_t KIndex1 = 0; KIndex1 < KWaveVectors.size(); KIndex1++)
    {

        for (size_t RIndex1 = 0; RIndex1 < RSites.size(); RIndex1++)
        {
            for (size_t RIndex2 = 0; RIndex2 < RSites.size(); RIndex2++)
            {

                greenK(KIndex1, KIndex1) += std::exp(-im * dot(KWaveVectors.at(KIndex1), RSites.at(RIndex1))) * std::exp(im * dot(KWaveVectors.at(KIndex1), RSites.at(RIndex2))) * greenR(RIndex1, RIndex2);
            }
        }
    }

    return (1.0 / static_cast<double>(RSites.size()) * greenK);
}

ClusterMatrixCD_t KtoR(const ClusterMatrixCD_t &greenK, const ClusterSites_t &RSites, const ClusterSites_t &KWaveVectors)
{
    assert(RSites.size() == KWaveVectors.size());
    ClusterMatrixCD_t greenR(KWaveVectors.size(), RSites.size());
    greenR.zeros();

    //for each matsubara freq.
    const cd_t im = cd_t(0.0, 1.0);

    for (size_t RIndex1 = 0; RIndex1 < RSites.size(); RIndex1++)
    {
        for (size_t RIndex2 = 0; RIndex2 < RSites.size(); RIndex2++)
        {
            for (size_t KIndex1 = 0; KIndex1 < KWaveVectors.size(); KIndex1++)
            {

                greenR(RIndex1, RIndex2) += std::exp(im * dot(KWaveVectors.at(KIndex1), RSites.at(RIndex1))) * std::exp(-im * dot(KWaveVectors.at(KIndex1), RSites.at(RIndex2))) * greenK(KIndex1, KIndex1);
            }
        }
    }
    return (1.0 / static_cast<double>(RSites.size()) * greenR);
}

} // namespace FourierDCA
#pragma once
#include "Utilities.hpp"
#include "GreenMat.hpp"

namespace Fourier
{

double MatToTau(const SiteVectorCD_t &greenMat, const double &tau, const double &beta) //Only for a "scalar green function, not a cluster green"
{
    double greenTau = 0.0;

    for (size_t n = 0; n < greenMat.size(); n++)
    {
        const double wn = (2.0 * n + 1.0) * M_PI / beta;
        greenTau += std::cos(wn * tau) * greenMat(n).real() + std::sin(wn * tau) * greenMat(n).imag();
    }

    return (2.0 * greenTau / beta);
}

double MatToTauAnalytic(SiteVectorCD_t greenMat, const double &tau, const double &beta, const double &fm, const double &sm, const double &tm)
{

    double result = 0.0;

    //result+= les moments en tau calculés analytiquement
    result += -0.5 * fm;
    result += (tau / 2.0 - beta / 4.0) * sm;
    result += -1.0 / 4.0 * (tau * (tau - beta)) * tm;

    //On transforme la greenMat moins ses moments
    for (size_t n = 0; n < greenMat.n_elem; n++)
    {
        const cd_t iwn(0.0, (2.0 * n + 1.0) * M_PI / beta);
        greenMat(n) -= fm / iwn + sm / (iwn * iwn) + tm / (iwn * iwn * iwn);
    }

    result += MatToTau(greenMat, tau, beta);

    return result;
}

ClusterMatrix_t MatToTauCluster(const GreenMat::GreenCluster0Mat &greenCluster0Mat, const double &tau)
{

    ClusterCubeCD_t dataMat = greenCluster0Mat.data();
    const double beta = greenCluster0Mat.beta();
    ClusterMatrixCD_t result(dataMat.n_rows, dataMat.n_cols);
    result.zeros();

    //result+= les moments en tau calculés analytiquement
    result += -0.5 * greenCluster0Mat.fm();
    result += (tau / 2.0 - beta / 4.0) * greenCluster0Mat.sm();
    result += -1.0 / 4.0 * (tau * (tau - beta)) * greenCluster0Mat.tm();

    //On transforme la greenMat moins ses moments
    for (size_t n = 0; n < dataMat.n_slices; n++)
    {
        cd_t wn(0.0, (2.0 * n + 1.0) * M_PI / beta);
        dataMat.slice(n) -= greenCluster0Mat.fm() / (wn) + greenCluster0Mat.sm() / (wn * wn) + greenCluster0Mat.tm() / (wn * wn * wn);
    }

    for (size_t i = 0; i < dataMat.n_rows; i++)
    {
        for (size_t j = 0; j < dataMat.n_cols; j++)
        {
            result(i, j) += MatToTau(dataMat.tube(i, j), tau, beta);
        }
    }

    ClusterMatrix_t resultReal(arma::real(result));
    return resultReal;
}

// ClusterMatrixCD_t KToR(const ClusterSitesCD_t &vectorK, const ClusterSites_t &KWaveVectors, const ClusterSites_t &RSites)
// {
//     const size_t NN = vectorK.n_elem;
//     assert(NN == Rsites.n_elem);
//     assert(NN == vectorK.n_elem);

//     ClusterMatrixCD_t matrixR(NN, NN);
//     matrixR.zeros();

//     for (size_t ii = 0; ii < NN; ii++)
//     {
//         for (size_t jj = 0; jj < NN; jj++)
//         {

//             const SiteVector_t Rdiff = RSites_[ii] - RSites_.at(jj);
//             for (size_t KK = 0; KK < NN; KK++)
//             {
//                 matrixR(ii, jj) += std::exp(im * dot(KWaveVectors(kk), Rdiff)) * vectorK(KK);
//             }
//         }
//     }

//     R /= static_cast<double>(NN);
//     return R;
// }

} // namespace Fourier
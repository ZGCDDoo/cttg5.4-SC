#pragma once

#include "../../Utilities/Utilities.hpp"
#include "../../Utilities/LinAlg.hpp"
#include "../ISData.hpp"
#include "../../Utilities/Matrix.hpp"

namespace Markov
{
namespace Obs
{

template <typename TIOModel, typename TModel>
class GreenBinning
{

  public:
    const size_t NAMBU_SIZE = 4;
    const size_t N_BIN_TAU = 10000;

    GreenBinning(const std::shared_ptr<TModel> &modelPtr, const std::shared_ptr<ISDataCT<TIOModel, TModel>> &dataCT,
                 const Json &jj, const FermionSpin_t &spin) : modelPtr_(modelPtr),
                                                              ioModel_(modelPtr_->ioModel()),
                                                              dataCT_(dataCT),
                                                              NMat_(0.5 * (jj["EGreen"].get<double>() * dataCT_->beta() / M_PI - 1.0)),
                                                              spin_(spin)
    {

        const size_t two = 2;
        const size_t LL = two * ioModel_.indepSites().size();

        M0Bins_.resize(NAMBU_SIZE, LL, N_BIN_TAU);
        M0Bins_.zeros();
        M1Bins_ = M0Bins_;
        M2Bins_ = M0Bins_;
        M3Bins_ = M0Bins_;
    }

    ClusterCubeCD_t greenNambuCube() const { return greenNambuCube_; };

    void MeasureGreenBinning(const Matrix<double> &Mmat)
    {

        const size_t N = dataCT_->vertices_.size();
        const double DeltaInv = N_BIN_TAU / dataCT_->beta_;
        if (N)
        {
            for (size_t p1 = 0; p1 < N; p1++)
            {
                for (size_t p2 = 0; p2 < N; p2++)
                {
                    const size_t s1 = dataCT_->vertices_.at(p1).site();
                    const size_t s2 = dataCT_->vertices_[p2].site();
                    const size_t ll = ioModel_.FindIndepSiteIndex(s1, s2);
                    const double factAnormal = ioModel_.SignFAnormal(s1, s2);
#ifdef AFM
                    SiteVector_t temp = static_cast<double>(dataCT_->sign_) * SiteVector_t({Mmat(2 * p1, 2 * p2), factAnormal * Mmat(2 * p1 + 1, 2 * p2), factAnormal * Mmat(2 * p1, 2 * p2 + 1), Mmat(2 * p1 + 1, 2 * p2 + 1)});
#else
                    const double MFanormal = 0.5 * factAnormal * (Mmat(2 * p1 + 1, 2 * p2) + Mmat(2 * p1, 2 * p2 + 1));
                    const double Mnormal = 0.5 * (Mmat(2 * p1, 2 * p2) + Mmat(2 * p1 + 1, 2 * p2 + 1));
                    SiteVector_t temp = static_cast<double>(dataCT_->sign_) * SiteVector_t({Mnormal, MFanormal, MFanormal, Mnormal});
#endif
                    double tau = dataCT_->vertices_[p1].tau() - dataCT_->vertices_[p2].tau();
                    if (tau < 0.0)
                    {
                        temp *= -1.0;
                        tau += dataCT_->beta_;
                    }

                    const size_t index = DeltaInv * tau;
                    const double dTau = tau - (static_cast<double>(index) + 0.5) / DeltaInv;

                    M0Bins_.slice(index).col(ll) += temp;
                    temp *= dTau;
                    M1Bins_.slice(index).col(ll) += temp;
                    temp *= dTau;
                    M2Bins_.slice(index).col(ll) += temp;
                    temp *= dTau;
                    M3Bins_.slice(index).col(ll) += temp;
                }
            }
        }
    }

    ClusterCubeCD_t FinalizeGreenBinning(const double &signMeas, const size_t &NMeas)
    {

        mpiUt::Print("Start of GreenBinning.FinalizeGreenBinning()");

        const double dTau = dataCT_->beta_ / N_BIN_TAU;
        const size_t LL = ioModel_.indepSites().size();
        ClusterMatrixCD_t indep_M_matsubara_sampled(NAMBU_SIZE, 2 * LL);
        SiteVectorCD_t temp_matsubara(NAMBU_SIZE);
        const ClusterCubeCD_t greenNambu0Cube = modelPtr_->nambuCluster0Mat().data();
        ClusterCubeCD_t greenNambuCube(2 * ioModel_.Nc, 2 * ioModel_.Nc, NMat_);
        greenNambuCube.zeros();
        // std::cout << "Here " << std::endl;

        for (size_t n = 0; n < NMat_; n++)
        {
            const double omega_n = M_PI * (2.0 * n + 1.0) / dataCT_->beta_;
            const cd_t iomega_n(0.0, omega_n);
            const cd_t fact = std::exp(iomega_n * dTau);
            const double lambda = 2.0 * std::sin(omega_n * dTau / 2.0) / (dTau * omega_n * (1.0 - omega_n * omega_n * dTau * dTau / 24.0) * NMeas);

            for (size_t ll = 0; ll < LL; ll++)
            {
                temp_matsubara.zeros();

                cd_t exp_factor = std::exp(iomega_n * dTau / 2.0) / (static_cast<double>(ioModel_.nOfAssociatedSites().at(ll))); //watch out important factor!
                for (size_t ii = 0; ii < N_BIN_TAU; ii++)
                {
                    const cd_t coeff = lambda * exp_factor;

                    temp_matsubara += coeff * M0Bins_.slice(ii).col(ll);
                    temp_matsubara += coeff * M1Bins_.slice(ii).col(ll) * iomega_n;
                    temp_matsubara += coeff * M2Bins_.slice(ii).col(ll) * iomega_n * iomega_n / 2.0;
                    temp_matsubara += coeff * M3Bins_.slice(ii).col(ll) * iomega_n * iomega_n * iomega_n / 6.0;

                    exp_factor *= fact;
                }
                indep_M_matsubara_sampled.col(ll) = temp_matsubara;
            }

            const ClusterMatrixCD_t dummy1 = ioModel_.IndepToFullNambu(indep_M_matsubara_sampled);
            const ClusterMatrixCD_t greenNambu0 = greenNambu0Cube.slice(n);

            greenNambuCube.slice(n) = greenNambu0 - greenNambu0 * dummy1 * greenNambu0 / (dataCT_->beta_ * signMeas);
            // std::cout << "Here 2 " << std::endl;
            if (n == 1)
            {
                dummy1.save("m01.dat", arma::arma_ascii);
            }
        }

        greenNambuCube_ = greenNambuCube; //in case it is needed later on

        // std::cout << "greenNambuCube.n_rows = " << greenNambuCube.n_rows << std::endl;
        // std::cout << "greenNambuCube.n_cols = " << greenNambuCube.n_cols << std::endl;
        // std::cout << "greenNambuCube.n_slices = " << greenNambuCube.n_slices << std::endl;

        mpiUt::Print("End of GreenBinning.FinalizeGreenBinning()");
        return greenNambuCube; //the  measured interacting green function
    }

  private:
    std::shared_ptr<TModel> modelPtr_;
    TIOModel ioModel_;
    std::shared_ptr<ISDataCT<TIOModel, TModel>> dataCT_;

    ClusterCube_t M0Bins_; //In Nambu style (first  indices is the nambu indices, second index is the indepsiteindex and last index (slice) is the time)
    ClusterCube_t M1Bins_;
    ClusterCube_t M2Bins_;
    ClusterCube_t M3Bins_;

    ClusterCubeCD_t greenNambuCube_;

    const size_t NMat_;
    const FermionSpin_t spin_;
};

} // namespace Obs
} // namespace Markov
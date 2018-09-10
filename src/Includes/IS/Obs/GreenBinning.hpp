#pragma once

#include "../../Utilities/Utilities.hpp"
#include "../../Utilities/LinAlg.hpp"
#include "../ISData.hpp"
#include "../../Utilities/Matrix.hpp"

namespace Markov
{
namespace Obs
{
const size_t N_BIN_TAU = 100000;

template <typename TIOModel, typename TModel>
class GreenBinning
{

  public:
    GreenBinning(const std::shared_ptr<TModel> &modelPtr, const std::shared_ptr<ISDataCT<TIOModel, TModel>> &dataCT,
                 const Json &jj, const FermionSpin_t &spin) : modelPtr_(modelPtr),
                                                              ioModel_(modelPtr_->ioModel()),
                                                              dataCT_(dataCT),
                                                              NMat_(0.5 * (jj["EGreen"].get<double>() * dataCT_->beta() / M_PI - 1.0)),
                                                              spin_(spin)
    {

        const size_t two = 2;
        const size_t LL = two * ioModel_.indepSites().size();

        M0Bins_.resize(4, LL, N_BIN_TAU);
        M0Bins_.zeros();
        M1Bins_ = M0Bins_;
        M2Bins_ = M0Bins_;
        M3Bins_ = M0Bins_;
    }

    ClusterCubeCD_t greenCube() const { return greenCube_; };

    void MeasureGreenBinning(const Matrix<double> &Mmat)
    {
        using arma::span;

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
                    SiteVector_t temp = static_cast<double>(dataCT_->sign_) * SiteVector_t({Mmat(2 * p1, 2 * p2), Mmat(2 * p1 + 1, 2 * p2), Mmat(2 * p1, 2 * p2 + 1), Mmat(2 * p1 + 1, 2 * p2 + 1)});

                    double tau = dataCT_->vertices_[p1].tau() - dataCT_->vertices_[p2].tau();
                    if (tau < 0.0)
                    {
                        temp *= -1.0;
                        tau += dataCT_->beta_;
                    }

                    const size_t index = DeltaInv * tau;
                    const double dTau = tau - (static_cast<double>(index) + 0.5) / DeltaInv;

                    M0Bins_(span(0, 4), span(ll, ll), span(index, index)) += temp;
                    temp *= dTau;
                    M1Bins_(span(0, 4), span(ll, ll), span(index, index)) += temp;
                    temp *= dTau;
                    M2Bins_(span(0, 4), span(ll, ll), span(index, index)) += temp;
                    temp *= dTau;
                    M3Bins_(span(0, 4), span(ll, ll), span(index, index)) += temp;
                }
            }
        }
    }

    ClusterCubeCD_t FinalizeGreenBinning(const double &signMeas, const size_t &NMeas)
    {
        using arma::span;

        mpiUt::Print("Start of GreenBinning.FinalizeGreenBinning()");

        const double dTau = dataCT_->beta_ / N_BIN_TAU;
        const size_t LL = ioModel_.indepSites().size();
        SiteVectorCD_t indep_M_matsubara_sampled(2 * LL);
        const ClusterCubeCD_t greenNambu0 = modelPtr_->greenNambu0();
        ClusterCubeCD_t greenNambu(2 * ioModel_.Nc, 2 * ioModel_.Nc, NMat_);
        greenNambu.zeros();

        for (size_t n = 0; n < NMat_; n++)
        {
            const double omega_n = M_PI * (2.0 * n + 1.0) / dataCT_->beta_;
            const cd_t iomega_n(0.0, omega_n);
            const cd_t fact = std::exp(iomega_n * dTau);
            const double lambda = 2.0 * std::sin(omega_n * dTau / 2.0) / (dTau * omega_n * (1.0 - omega_n * omega_n * dTau * dTau / 24.0) * NMeas);

            for (size_t ll = 0; ll < LL; ll++)
            {
                SiteVectorCD_t temp_matsubara(2 * LL);
                temp_matsubara.zeros();

                cd_t exp_factor = std::exp(iomega_n * dTau / 2.0) / (static_cast<double>(ioModel_.nOfAssociatedSites().at(ll))); //watch out important factor!
                for (size_t ii = 0; ii < N_BIN_TAU; ii++)
                {
                    const cd_t coeff = lambda * exp_factor;

                    temp_matsubara += coeff * M0Bins_(span(0, 4), span(ll, ll), span(ii, ii));
                    temp_matsubara += coeff * M1Bins_(span(0, 4), span(ll, ll), span(ii, ii)) * iomega_n;
                    temp_matsubara += coeff * M2Bins_(span(0, 4), span(ll, ll), span(ii, ii)) * iomega_n * iomega_n / 2.0;
                    temp_matsubara += coeff * M3Bins_(span(0, 4), span(ll, ll), span(ii, ii)) * iomega_n * iomega_n * iomega_n / 6.0;

                    exp_factor *= fact;
                }
                indep_M_matsubara_sampled(ll) = temp_matsubara;
            }

            //     const ClusterMatrixCD_t dummy1 = ioModel_.IndepToFull(indep_M_matsubara_sampled);
            //     const ClusterMatrixCD_t green0 = green0CubeMatsubara.slice(n);

            //     greenCube.slice(n) = green0 - green0 * dummy1 * green0 / (dataCT_->beta_ * signMeas);
        }

        // greenCube_ = greenCube; //in case it is needed later on

        // mpiUt::Print("End of GreenBinning.FinalizeGreenBinning()");
        // return greenCube; //the  measured interacting green function
    }

  private:
    std::shared_ptr<TModel> modelPtr_;
    TIOModel ioModel_;
    std::shared_ptr<ISDataCT<TIOModel, TModel>> dataCT_;

    ClusterCube_t M0Bins_; //In Nambu style (first  indices is the nambu indices, second index is the indepsiteindex and last index (slice) is the time)
    ClusterCube_t M1Bins_;
    ClusterCube_t M2Bins_;
    ClusterCube_t M3Bins_;

    ClusterCubeCD_t greenCube_;

    const size_t NMat_;
    const FermionSpin_t spin_;
};

} // namespace Obs
} // namespace Markov
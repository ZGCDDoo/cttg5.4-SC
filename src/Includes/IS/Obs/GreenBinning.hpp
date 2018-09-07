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

        const size_t LL = ioModel_.indepSites().size();
        M0Bins_.resize(LL);
        M1Bins_.resize(LL);
        M2Bins_.resize(LL);
        M3Bins_.resize(LL);

        for (size_t ii = 0; ii < M0Bins_.size(); ii++)
        {
            M0Bins_.at(ii).resize(N_BIN_TAU, 0.0);
            M1Bins_.at(ii).resize(N_BIN_TAU, 0.0);
            M2Bins_.at(ii).resize(N_BIN_TAU, 0.0);
            M3Bins_.at(ii).resize(N_BIN_TAU, 0.0);
        }
    }

    ClusterCubeCD_t greenCube() const { return greenCube_; };

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
                    const size_t s1 = dataCT_->vertices_[p1].site();
                    const size_t s2 = dataCT_->vertices_[p2].site();
                    const size_t ll = ioModel_.FindIndepSiteIndex(s1, s2);
                    double temp = static_cast<double>(dataCT_->sign_) * Mmat(p1, p2);

                    double tau = dataCT_->vertices_[p1].tau() - dataCT_->vertices_[p2].tau();
                    if (tau < 0.0)
                    {
                        temp *= -1.0;
                        tau += dataCT_->beta_;
                    }

                    const int index = DeltaInv * tau;
                    const double dTau = tau - (static_cast<double>(index) + 0.5) / DeltaInv;

                    M0Bins_.at(ll).at(index) += temp;
                    temp *= dTau;
                    M1Bins_[ll][index] += temp;
                    temp *= dTau;
                    M2Bins_[ll][index] += temp;
                    temp *= dTau;
                    M3Bins_[ll][index] += temp;
                }
            }
        }
    }

    ClusterCubeCD_t FinalizeGreenBinning(const double &signMeas, const size_t &NMeas)
    {
        mpiUt::Print("Start of GreenBinning.FinalizeGreenBinning()");

        const double dTau = dataCT_->beta_ / N_BIN_TAU;
        SiteVectorCD_t indep_M_matsubara_sampled(ioModel_.indepSites().size());
        const ClusterCubeCD_t green0CubeMatsubara = spin_ == FermionSpin_t::Up ? modelPtr_->greenCluster0MatUp().data() : modelPtr_->greenCluster0MatDown().data();
        ClusterCubeCD_t greenCube(ioModel_.Nc, ioModel_.Nc, NMat_);
        greenCube.zeros();

        for (size_t n = 0; n < NMat_; n++)
        {
            const double omega_n = M_PI * (2.0 * n + 1.0) / dataCT_->beta_;
            const cd_t iomega_n(0.0, omega_n);
            const cd_t fact = std::exp(iomega_n * dTau);
            const double lambda = 2.0 * std::sin(omega_n * dTau / 2.0) / (dTau * omega_n * (1.0 - omega_n * omega_n * dTau * dTau / 24.0) * NMeas);

            for (size_t ll = 0; ll < ioModel_.indepSites().size(); ll++)
            {
                cd_t temp_matsubara = 0.0;

                cd_t exp_factor = std::exp(iomega_n * dTau / 2.0) / (static_cast<double>(ioModel_.nOfAssociatedSites().at(ll))); //watch out important factor!
                for (size_t ii = 0; ii < N_BIN_TAU; ii++)
                {
                    cd_t coeff = lambda * exp_factor;

                    temp_matsubara += coeff * M0Bins_.at(ll).at(ii);
                    temp_matsubara += coeff * M1Bins_[ll][ii] * iomega_n;
                    temp_matsubara += coeff * M2Bins_[ll][ii] * iomega_n * iomega_n / 2.0;
                    temp_matsubara += coeff * M3Bins_[ll][ii] * iomega_n * iomega_n * iomega_n / 6.0;

                    exp_factor *= fact;
                }
                indep_M_matsubara_sampled(ll) = temp_matsubara;
            }

            const ClusterMatrixCD_t dummy1 = ioModel_.IndepToFull(indep_M_matsubara_sampled);
            const ClusterMatrixCD_t green0 = green0CubeMatsubara.slice(n);

            greenCube.slice(n) = green0 - green0 * dummy1 * green0 / (dataCT_->beta_ * signMeas);
        }

        greenCube_ = greenCube; //in case it is needed later on

        mpiUt::Print("End of GreenBinning.FinalizeGreenBinning()");
        return greenCube; //the  measured interacting green function
    }

  private:
    std::shared_ptr<TModel> modelPtr_;
    TIOModel ioModel_;
    std::shared_ptr<ISDataCT<TIOModel, TModel>> dataCT_;

    std::vector<std::vector<double>> M0Bins_;
    std::vector<std::vector<double>> M1Bins_;
    std::vector<std::vector<double>> M2Bins_;
    std::vector<std::vector<double>> M3Bins_;

    ClusterCubeCD_t greenCube_;

    const size_t NMat_;
    const FermionSpin_t spin_;
};

} // namespace Obs
} // namespace Markov
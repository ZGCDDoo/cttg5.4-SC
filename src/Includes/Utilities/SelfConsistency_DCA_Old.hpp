#pragma once

#include "Integrator.hpp"
#include "Utilities.hpp"
#include "MPIUtilities.hpp"
#include "GreenMat.hpp"
#include "ABC_SelfConsistency.hpp"
#include "Fourier_DCA.hpp"

namespace SelfCon
{

template <typename TIOModel, typename TModel, typename TH0>
class SelfConsistency : public ABC_SelfConsistency
{

  public:
    static const size_t Nc;
    static const ClusterMatrixCD_t II;
    static const double factNSelfCon;

    SelfConsistency(const Json &jj, const TModel &model, const ClusterCubeCD_t &greenImpurity, const std::string &spinName) : model_(model),
                                                                                                                              ioModel_(TIOModel()),
                                                                                                                              h0_(jj["t"].get<double>(), jj["tPrime"].get<double>(), jj["tPrimePrime"].get<double>()),
                                                                                                                              greenImpurityK_(FourierDCA::RtoK(greenImpurity, h0_.RSites(), h0_.KWaveVectors())),
                                                                                                                              hybridizationK_(model_.hybridizationKMatUp()),
                                                                                                                              selfEnergyK_(),
                                                                                                                              spinName_(spinName),
                                                                                                                              weights_(cd_t(jj["WEIGHTSR"].get<double>(), jj["WEIGHTSI"].get<double>()))
    {

        mpiUt::Print("Start of SC constructor");
        const size_t NGreen = greenImpurity.n_slices;
        const size_t NSelfCon = NGreen;
        assert(NSelfCon >= NGreen);
        //Patcher la hyb si necessaire
        hybridizationK_.PatchHF(NSelfCon, model_.beta());
        const size_t NHyb = hybridizationK_.n_slices();
        assert(NHyb >= NSelfCon);

        selfEnergyK_.resize(Nc, Nc, NSelfCon);
        selfEnergyK_.zeros();

        //0.) Extraire la self jusqu'a NGreen
        for (size_t nn = 0; nn < NGreen; nn++)
        {
            const cd_t zz = cd_t(model_.mu(), (2.0 * nn + 1.0) * M_PI / model_.beta());
            selfEnergyK_.slice(nn) = -greenImpurityK_.slice(nn).i() + zz * ClusterMatrixCD_t(Nc, Nc).eye() - model_.epsKBar() - hybridizationK_.slice(nn);
        }

        //1.) Patcher la self par HF de NGreen à NSelfCon
        ClusterMatrix_t nUpMatrixK;
        assert(nUpMatrixK.load("nUpMatrix.dat"));
        ClusterMatrix_t nDownMatrixK;
        assert(nDownMatrixK.load("nDownMatrix.dat"));
        ClusterMatrixCD_t nMatrixK(nUpMatrixK + nDownMatrixK, ClusterMatrix_t(Nc, Nc).zeros());
        nMatrixK = FourierDCA::RtoK(nMatrixK, h0_.RSites(), h0_.KWaveVectors());

        // for (size_t nn = NGreen; nn < NSelfCon; nn++)
        // {
        //     cd_t iwn = cd_t(0.0, (2.0 * nn + 1.0) * M_PI / model_.beta());
        //     selfEnergyK_.slice(nn) = 0.5 * model_.U() * nMatrixK + 1.0 / iwn * model_.U() * model_.U() * nMatrixK / 2.0 * (II - nMatrixK / 2.0);
        // }

        if (mpiUt::Rank() == mpiUt::master)
        {
            SaveK("self" + spinName_, selfEnergyK_);
        }

        mpiUt::Print("After SC constructor");
    }

    void DoSCGrid() override
    {

        DoSCGridSerial();
    }

    void DoSCGridSerial()
    {

        if (mpiUt::Rank() == mpiUt::master)
        {
            std::cout << "In Selfonsistency DOSC serial" << std::endl;
            const size_t NSelfCon = selfEnergyK_.n_slices;
            const size_t NKPTS = 100;
            ClusterCubeCD_t gImpUpKNext(Nc, Nc, NSelfCon);
            assert(Nc == h0_.KWaveVectors().size());
            gImpUpKNext.zeros();
            hybKNext_ = gImpUpKNext;

            const double kxCenter = M_PI / static_cast<double>(h0_.Nx);
            const double kyCenter = M_PI / static_cast<double>(h0_.Ny);

            for (size_t KIndex = 0; KIndex < h0_.KWaveVectors().size(); KIndex++)
            {
                const double Kx = h0_.KWaveVectors().at(KIndex)(0);
                const double Ky = h0_.KWaveVectors().at(KIndex)(1);
                for (size_t nn = 0; nn < NSelfCon; nn++)
                {
                    const cd_t zz = cd_t(model_.mu(), (2.0 * nn + 1.0) * M_PI / model_.beta());
                    for (size_t kxindex = 0; kxindex < NKPTS; kxindex++)
                    {
                        const double kx = (Kx - kxCenter) + static_cast<double>(kxindex) / static_cast<double>(NKPTS) * 2.0 * kxCenter;
                        for (size_t kyindex = 0; kyindex < NKPTS; kyindex++)
                        {
                            const double ky = (Ky - kyCenter) + static_cast<double>(kyindex) / static_cast<double>(NKPTS) * 2.0 * kyCenter;
                            gImpUpKNext(KIndex, KIndex, nn) += 1.0 / (zz - h0_.Eps0k(kx, ky) - selfEnergyK_(KIndex, KIndex, nn));
                        }
                    }
                    gImpUpKNext(KIndex, KIndex, nn) *= 1.0 / static_cast<double>(NKPTS * NKPTS);
                }
            }

            SaveK("green" + spinName_, gImpUpKNext);

            for (size_t nn = 0; nn < gImpUpKNext.n_slices; nn++)
            {
                const cd_t zz = cd_t(model_.mu(), (2.0 * nn + 1.0) * M_PI / model_.beta());
                hybKNext_.slice(nn) = -gImpUpKNext.slice(nn).i() - selfEnergyK_.slice(nn) + zz * ClusterMatrixCD_t(Nc, Nc).eye() - model_.epsKBar();
            }

            SaveK("hybNext" + spinName_, hybKNext_);

            std::cout << "After Selfonsistency DOSC serial" << std::endl;
        }
    }

    void SaveK(std::string fname, FourierDCA::DataK_t greenK)
    {

        std::ofstream fout;
        fout.open(fname + std::string(".dat"), std::ios::out);

        for (size_t nn = 0; nn < greenK.n_slices; nn++)
        {
            const double iwn = (2.0 * nn + 1.0) * M_PI / model_.beta();
            fout << iwn << " ";

            for (size_t KIndex = 0; KIndex < h0_.KWaveVectors().size(); KIndex++)
            {

                fout << greenK(KIndex, KIndex, nn).real()
                     << " "
                     << greenK(KIndex, KIndex, nn).imag()
                     << " ";
            }
            fout << "\n";
        }

        fout.close();
    }

  private:
    TModel model_;
    TIOModel ioModel_;
    TH0 h0_;

    const ClusterCubeCD_t greenImpurityK_;
    GreenMat::HybridizationMat hybridizationK_;

    FourierDCA::DataK_t selfEnergyK_;
    FourierDCA::DataK_t hybKNext_;

    const std::string spinName_;
    const cd_t weights_;
};
template <typename TIOModel, typename TModel, typename TH0>
const ClusterMatrixCD_t SelfConsistency<TIOModel, TModel, TH0>::II = ClusterMatrixCD_t(TH0::Nc, TH0::Nc).eye();

template <typename TIOModel, typename TModel, typename TH0>
const size_t SelfConsistency<TIOModel, TModel, TH0>::Nc = TModel::Nc;

template <typename TIOModel, typename TModel, typename TH0>
const double SelfConsistency<TIOModel, TModel, TH0>::factNSelfCon = 2;

} // namespace SelfCon
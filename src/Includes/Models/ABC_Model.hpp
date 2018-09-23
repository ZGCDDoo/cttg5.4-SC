#pragma once

#include "../Utilities/Utilities.hpp"
#include "../Utilities/MPIUtilities.hpp"
#include "../Utilities/NambuMat.hpp"
#include "../Utilities/IO.hpp"
#include "HybFMAndTLoc.hpp"
// #include "../Utilities/Conventions.hpp"

using Vertex = Utilities::Vertex;

namespace Models
{

const size_t Nx1 = 1;
const size_t Nx2 = 2;
const size_t Nx4 = 4;
const size_t Nx6 = 6;
const size_t Nx8 = 8;

template <typename TIOModel, typename TH0> //Template Number of X, Y sites
class ABC_Model_2D
{

      public:
        static const size_t Nc;

        ABC_Model_2D(const Json &jj) : ioModel_(TIOModel()),
                                       h0_(jj["t"].get<double>(), jj["tPrime"].get<double>(), jj["tPrimePrime"].get<double>()),
                                       hybFM_(),
                                       tLoc_(),
                                       U_(jj["U"].get<double>()),
                                       delta_(jj["delta"].get<double>()),
                                       beta_(jj["beta"].get<double>()),
                                       mu_(jj["mu"].get<double>()),
                                       K_(jj["K"].get<double>()),
                                       gamma_(std::acosh(1.0 + U_ * beta_ * TH0::Nc / (2.0 * K_)))
        {
                mpiUt::Print("start abc_model constructor ");
                if (mpiUt::Rank() == mpiUt::master)
                {

#ifdef DCA
                        HybFMAndTLoc<TH0>::CalculateHybFMAndTLoc(h0_);
#else
                        h0_.SaveTKTildeAndHybFM();
#endif
                }

//tLoc and hybFM should have been calculated by now.
#ifdef DCA
                assert(tLoc_.load("tloc_K.arma"));
                assert(hybFM_.load("hybFM_K.arma"));
#else
                assert(tLoc_.load("tloc.arma"));
                assert(hybFM_.load("hybFM.arma"));
#endif
                FinishConstructor(jj);
                mpiUt::Print(" End of ABC_Model Constructor ");
        };

        void FinishConstructor(const Json &jj)
        {

                ClusterCubeCD_t hybNambuData;
                ClusterMatrixCD_t hybSM;
                if (!hybNambuData.load("hybNextNambu.arma"))
                {
                        hybNambuData.resize(2 * Nc, 2 * Nc, 1);
                        hybNambuData.slice(0) = ClusterMatrixCD_t(2 * Nc, 2 * Nc).eye() / (cd_t(0.0, M_PI / beta_));
                        // const ClusterMatrix_t II2x2Off = {{0.0, 1.0}, {1.0, 0.0}};
                        // hybSM = arma::kron(II2x2Off, 1e-4 * ioModel_.signFAnormal());
                }

                if (jj["BREAK_SYMMETRY"].get<bool>() == true)
                {
                        for (size_t ii = 0; ii < Nc; ii++)
                        {
                                for (size_t jj = 0; jj < Nc; jj++)
                                {
                                        hybNambuData(ii, jj + Nc, 0) = 1e-1 * ioModel_.SignFAnormal(ii, jj);
                                        hybNambuData(ii + Nc, jj, 0) = hybNambuData(ii, jj + Nc, 0);
                                }
                        }
                }

                const size_t NHyb = hybNambuData.n_slices;
                const double factNHyb = 3.0;
                const size_t NHyb_HF = std::max<double>(factNHyb * static_cast<double>(NHyb),
                                                        0.5 * (300.0 * beta_ / M_PI - 1.0));

                hybridizationMat_ = NambuMat::HybridizationMat(hybNambuData, this->hybFM_, hybSM);

                hybridizationMat_.PatchHF(NHyb_HF, beta_);

                const size_t NNambu = 2 * Nc;
                const ClusterMatrixCD_t II2x2 = ClusterMatrixCD_t(2, 2).eye();
                const ClusterMatrixCD_t II2x2Nambu = {{cd_t(1.0), cd_t(0.0)}, {cd_t(0.0), cd_t(-1.0)}};

                const ClusterMatrixCD_t II = ClusterMatrixCD_t(NNambu, NNambu).eye();
                const ClusterMatrixCD_t IINambu = arma::kron(II2x2Nambu, ClusterMatrixCD_t(Nc, Nc).eye());

                const ClusterMatrixCD_t II2x2_10 = {{cd_t(1.0), cd_t(0.0)}, {cd_t(0.0), cd_t(0.0)}};
                const ClusterMatrixCD_t II_10 = arma::kron(II2x2_10, ClusterMatrixCD_t(Nc, Nc).eye());

                const ClusterMatrixCD_t muNambu = mu_ * IINambu - U_ * II_10 + U_ / 2.0 * II;

                nambuCluster0Mat_ = NambuMat::NambuCluster0Mat(hybridizationMat_, tLoc_, muNambu, beta_);
        }

        virtual ~ABC_Model_2D() = 0;

        //Getters
        double mu() const { return mu_; };
        double U() const { return U_; };
        double delta() const { return delta_; };
        double beta() const { return beta_; };
        ClusterMatrixCD_t tLoc() const { return tLoc_; };

        NambuMat::NambuCluster0Mat const nambuCluster0Mat() { return nambuCluster0Mat_; };
        NambuMat::HybridizationMat const hybridizationMat() const { return hybridizationMat_; };

        TH0 const h0() { return h0_; };
        TIOModel const ioModel() { return ioModel_; };

        //Maybe put everything concerning aux spins in vertex class. therfore delta in vertex constructor.
        double auxUp(const AuxSpin_t &aux) const { return ((aux == AuxSpin_t::Up) ? 1.0 + delta_ : -delta_); };
        double auxDown(const AuxSpin_t &aux) const { return ((aux == AuxSpin_t::Up) ? 1.0 + delta_ : -delta_); };

        double FAuxUp(const AuxSpin_t &aux) const
        {
                if (aux == AuxSpin_t::Zero)
                {
                        return 1.0;
                }
                return (auxUp(aux) / (auxUp(aux) - 1.0));
        };

        double FAuxDown(const AuxSpin_t &aux) const
        {
                if (aux == AuxSpin_t::Zero)
                {
                        return 1.0;
                }
                return (auxDown(aux) / (auxDown(aux) - 1.0));
        };

        double gammaUp(const AuxSpin_t &auxI, const AuxSpin_t &auxJ) const //little gamma
        {
                const double fsJ = FAuxUp(auxJ);
                return ((FAuxUp(auxI) - fsJ) / fsJ);
        }

        double gammaDown(const AuxSpin_t &auxI, const AuxSpin_t &auxJ) const //little gamma
        {
                const double fsJ = FAuxDown(auxJ);
                return ((FAuxDown(auxI) - fsJ) / fsJ);
        }

        double KAux(const AuxSpin_t &aux) const
        {
                return (U_ * beta_ * Nc / ((FAuxUp(aux) - 1.0) * (FAuxDown(aux) - 1.0)));
        }

        double K() const { return K_; };
        double gamma() const { return gamma_; };

      protected:
        TIOModel ioModel_;

        NambuMat::HybridizationMat hybridizationMat_;
        NambuMat::NambuCluster0Mat nambuCluster0Mat_;

        TH0 h0_;

        ClusterMatrixCD_t hybFM_;
        ClusterMatrixCD_t tLoc_;

        const double U_;
        const double delta_;
        const double beta_;
        double mu_;
        const double K_;
        const double gamma_;
};

template <typename TIOModel, typename TH0>
ABC_Model_2D<TIOModel, TH0>::~ABC_Model_2D() {} //destructors must exist

template <typename TIOModel, typename TH0>
const size_t ABC_Model_2D<TIOModel, TH0>::Nc = TH0::Nc;

} // namespace Models

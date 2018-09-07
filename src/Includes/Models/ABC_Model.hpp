#pragma once

#include "../Utilities/Utilities.hpp"
#include "../Utilities/MPIUtilities.hpp"
#include "../Utilities/Integrator.hpp"
#include "../Utilities/GreenMat.hpp"
#include "../Utilities/IO.hpp"
#include "HybFMAndTLoc.hpp"

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
                std::string hybNameUp = jj["HybFileUp"].get<std::string>();
#ifdef DCA
                ClusterCubeCD_t hybtmpUp = ioModel_.ReadGreenKDat(hybNameUp + ".dat");
#else
                ClusterCubeCD_t hybtmpUp = ioModel_.ReadGreenDat(hybNameUp + ".dat");
#endif

#ifdef AFM
                std::string hybNameDown = jj["HybFileDown"].get<std::string>();
                ClusterCubeCD_t hybtmpDown = ioModel_.ReadGreenDat(hybNameDown + ".dat");
#endif

                const size_t NHyb = hybtmpUp.n_slices;
                const double factNHyb = 3.0;
                const size_t NHyb_HF = std::max<double>(factNHyb * static_cast<double>(NHyb),
                                                        0.5 * (300.0 * beta_ / M_PI - 1.0));
                hybtmpUp.resize(Nc, Nc, NHyb_HF);
#ifdef AFM
                hybtmpDown.resize(Nc, Nc, NHyb_HF);
#endif

                for (size_t nn = NHyb; nn < NHyb_HF; nn++)
                {
                        cd_t iwn(0.0, (2.0 * nn + 1.0) * M_PI / beta_);
                        hybtmpUp.slice(nn) = hybFM_ / iwn;
#ifdef AFM
                        hybtmpDown.slice(nn) = hybtmpUp.slice(nn);
#endif
                }

                this->hybridizationMatUp_ = GreenMat::HybridizationMat(hybtmpUp, this->hybFM_);
#ifdef AFM
                this->hybridizationMatDown_ = GreenMat::HybridizationMat(hybtmpDown, this->hybFM_);
#endif

                //this is in fact greencluster tilde.
                this->greenCluster0MatUp_ = GreenMat::GreenCluster0Mat(this->hybridizationMatUp_, this->tLoc_, this->auxMu(), this->beta_);
#ifdef DCA
                greenCluster0MatUp_.FourierTransform(h0_.RSites(), h0_.KWaveVectors());
#endif
                //save green0mat
                if (mpiUt::Rank() == mpiUt::master)
                {
                        ioModel_.SaveCube("giwn", this->greenCluster0MatUp_.data(), this->beta_);
                }
#ifndef AFM
                this->greenCluster0MatDown_ = greenCluster0MatUp_;
#endif
#ifdef AFM
                this->greenCluster0MatDown_ = GreenMat::GreenCluster0Mat(this->hybridizationMatDown_, this->tLoc_, this->auxMu(), this->beta_);
#endif
        }

        virtual ~ABC_Model_2D() = 0;

        //Getters
        double mu() const { return mu_; };
        double U() const { return U_; };
        double delta() const { return delta_; };
        double beta() const { return beta_; };
        ClusterMatrixCD_t tLoc() const { return tLoc_; };
        GreenMat::GreenCluster0Mat const greenCluster0MatUp() { return greenCluster0MatUp_; };
        GreenMat::GreenCluster0Mat const greenCluster0MatDown() { return greenCluster0MatDown_; };
        GreenMat::HybridizationMat const hybridizationMatUp() const { return hybridizationMatUp_; };
        GreenMat::HybridizationMat const hybridizationMatDown() const { return hybridizationMatDown_; };
        TH0 const h0() { return h0_; };
        TIOModel const ioModel() { return ioModel_; };

        //Maybe put everything concerning aux spins in vertex class. therfore delta in vertex constructor.
        double auxUp(const AuxSpin_t &aux) const { return ((aux == AuxSpin_t::Up) ? 1.0 + delta_ : -delta_); };
        double auxDown(const AuxSpin_t &aux) const { return ((aux == AuxSpin_t::Down) ? 1.0 + delta_ : -delta_); };

        double FAuxUp(const AuxSpin_t &aux)
        {
                if (aux == AuxSpin_t::Zero)
                {
                        return 1.0;
                }
                return (auxUp(aux) / (auxUp(aux) - 1.0));
        };

        double FAuxDown(const AuxSpin_t &aux)
        {
                if (aux == AuxSpin_t::Zero)
                {
                        return 1.0;
                }
                return (auxDown(aux) / (auxDown(aux) - 1.0));
        };

        double gammaUp(const AuxSpin_t &auxI, const AuxSpin_t &auxJ) //little gamma
        {
                double fsJ = FAuxUp(auxJ);
                return ((FAuxUp(auxI) - fsJ) / fsJ);
        }

        double gammaDown(const AuxSpin_t &auxI, const AuxSpin_t &auxJ) //little gamma
        {
                double fsJ = FAuxDown(auxJ);
                return ((FAuxDown(auxI) - fsJ) / fsJ);
        }

        double KAux()
        {
                return (-U_ * beta_ * Nc / (((1.0 + delta_) / delta_ - 1.0) * (delta_ / (1.0 + delta_) - 1.0)));
        }

        double auxU() const { return U_ / 2.0; };
        double auxMu() const { return mu_ - U_ / 2.0; };
        double auxDO() const { return delta_ * (1.0 + delta_); };
        double K() const { return K_; };
        double gamma() const { return gamma_; };

      protected:
        TIOModel ioModel_;
        GreenMat::HybridizationMat hybridizationMatUp_;
        GreenMat::HybridizationMat hybridizationMatDown_;
        GreenMat::GreenCluster0Mat greenCluster0MatUp_;
        GreenMat::GreenCluster0Mat greenCluster0MatDown_;
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

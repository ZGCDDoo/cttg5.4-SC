#pragma once

#include "../Utilities/Utilities.hpp"
#include "../Utilities/LinAlg.hpp"
#include "../Utilities/MPIUtilities.hpp"
#include "../Utilities/Fourier.hpp"
#include "../Utilities/GreenTau.hpp"
#include "Obs/Observables.hpp"
#include "ISData.hpp"

//We have the conig C0TIlde = C0 U CTilde
//C0 is only the interacting vertices.
//ctilde is only the non-interacting vertices (before insertions are proposed)

namespace Markov
{

using Fourier::MatToTau;
using Fourier::MatToTauCluster;
using Vertex = Utilities::Vertex;
typedef LinAlg::Matrix_t Matrix_t;

extern "C"
{
    unsigned int dcopy_(unsigned int const *, double const *, unsigned int const *, double *, unsigned int const *);
}

struct GammaData
{
    GammaData() : gammaUpI_(), gammaDownI_(){};
    Matrix_t gammaUpI_;
    Matrix_t gammaDownI_;
};

struct UpdData
{
    UpdData() : xup_(), yup_(), xdown_(), ydown_(){};

    SiteVector_t xup_; //New row to insert
    SiteVector_t yup_; //New column to insert
    SiteVector_t xdown_;
    SiteVector_t ydown_;

    SiteVector_t gammaUpIYup_; // gammaUpI_*yup;
    SiteVector_t gammaDownIYdown_;
    double dup_;
    double ddown_;
    double dTildeUpI_;
    double dTildeDownI_;

    void SetSize(const size_t &kk)
    {
        xup_.set_size(kk);
        yup_.set_size(kk);
        xdown_.set_size(kk);
        ydown_.set_size(kk);
        gammaUpIYup_.set_size(kk);
        gammaDownIYdown_.set_size(kk);
    }
};

struct NFData
{

    NFData() : FVup_(), FVdown_(), Nup_(), Ndown_(){};
    SiteVector_t FVup_;
    SiteVector_t FVdown_;
    Matrix_t Nup_;
    Matrix_t Ndown_;
};

struct GreenData
{

    GreenData() : greenInteractUp_(), greenInteractDown_(){};
    Matrix_t greenInteractUp_;
    Matrix_t greenInteractDown_;
};

template <typename TIOModel, typename TModel>
class ABC_MarkovChainSubMatrix
{

    using GreenTau_t = GreenTau::GreenCluster0Tau<TIOModel>;

  public:
    const size_t Nc = TModel::Nc;
    const double PROBINSERT = 0.5;
    const double PROBREMOVE = 1.0 - PROBINSERT;

    ABC_MarkovChainSubMatrix(const Json &jj, const size_t &seed) : model_(jj),
                                                                   rng_(seed),
                                                                   urng_(rng_, Utilities::UniformDistribution_t(0.0, 1.0)),
                                                                   gammadata_(),
                                                                   upddata_(),
                                                                   nfdata_(),
                                                                   greendata_(),
                                                                   dataCT_(
                                                                       new Obs::ISDataCT<TIOModel, TModel>(
                                                                           static_cast<double>(jj["beta"].get<double>()),
                                                                           model_, jj["NTAU"].get<double>())),
                                                                   obs_(dataCT_, jj),
                                                                   KMAX_UPD_(jj["KMAX_UPD"].get<double>()),
                                                                   expUp_(std::exp(model_.gamma())),
                                                                   expDown_(std::exp(-model_.gamma()))
    {
        const std::valarray<size_t> zeroPair = {0, 0};
        updStats_["Inserts"] = zeroPair;
        updStats_["Removes"] = zeroPair;
        updStats_["Flips"] = zeroPair;

        updatesProposed_ = 0;

        mpiUt::Print("MarkovChain Created \n");
    }

    virtual ~ABC_MarkovChainSubMatrix() = 0;

    //Getters
    TModel model() const
    {
        return model_;
    };
    Matrix_t Nup() const
    {
        return nfdata_.Nup_;
    };
    Matrix_t Ndown() const
    {
        return nfdata_.Ndown_;
    };
    std::vector<Vertex> vertices() const
    {
        return dataCT_->vertices_;
    };
    double beta() const
    {
        return dataCT_->beta_;
    };

    size_t updatesProposed() const
    {
        return updatesProposed_;
    }

    void AssertSizes()
    {
        const size_t kk = dataCT_->vertices_.size();
        assert(kk == nfdata_.Nup_.n_rows());
        assert(kk == nfdata_.Nup_.n_cols());
        assert(kk == nfdata_.Ndown_.n_rows());
        assert(kk == nfdata_.Ndown_.n_cols());
    }

    virtual double gammaUpSubMatrix(const AuxSpin_t &auxxTo, const AuxSpin_t &vauxFrom) = 0;
    virtual double gammaDownSubMatrix(const AuxSpin_t &auxxTo, const AuxSpin_t &vauxFrom) = 0;
    virtual double KAux() = 0;
    virtual double FAuxUp(const AuxSpin_t &aux) = 0;
    virtual double FAuxDown(const AuxSpin_t &aux) = 0;

    void ThermalizeFromConfig()
    {
        if (mpiUt::LoadConfig(dataCT_->vertices_))
        {
            const size_t kk = dataCT_->vertices_.size();
            nfdata_.FVup_ = SiteVector_t(kk);
            nfdata_.FVdown_ = SiteVector_t(kk);
            for (size_t i = 0; i < kk; i++)
            {
                AuxSpin_t aux = dataCT_->vertices_[i].aux();
                nfdata_.FVup_(i) = FAuxUp(aux);
                nfdata_.FVdown_(i) = FAuxDown(aux);
            }

            nfdata_.Nup_.Resize(kk, kk);
            nfdata_.Ndown_.Resize(kk, kk);
            CleanUpdate();
        }
    }

    void DoStep()
    {
        // std::cout << "In do Step" << std::endl;
        PreparationSteps();

        for (size_t kk = 0; kk < KMAX_UPD_; kk++)
        {
            DoInnerStep();
            updatesProposed_++;
        }

        UpdateSteps();
    }

    void DoInnerStep()
    {

        // if (urng_() < PROBFLIP_)
        // {
        //     FlipAuxSubMatrix();
        // }
        // else
        // {
        urng_() < PROBINSERT ? InsertVertexSubMatrix() : RemoveVertexSubMatrix();
        // }
    }

    double CalculateDeterminantRatio(const Vertex &vertexTo, const Vertex &vertexFrom, const size_t &vertexIndex)
    {
        // std::cout << "in calculate determinant " << std::endl;

        AuxSpin_t auxTo = vertexTo.aux();
        AuxSpin_t auxFrom = vertexFrom.aux();
        double ratio;
        double gammakup = gammaUpSubMatrix(auxTo, auxFrom);
        double gammakdown = gammaDownSubMatrix(auxTo, auxFrom);
        upddata_.dup_ = (greendata_.greenInteractUp_(vertexIndex, vertexIndex) - (1.0 + gammakup) / gammakup);
        upddata_.ddown_ = (greendata_.greenInteractDown_(vertexIndex, vertexIndex) - (1.0 + gammakdown) / gammakdown);

        // std::cout << " calculating gamma = " << gammakup << std::endl;
        if (gammadata_.gammaUpI_.n_rows())
        {
            const size_t kkold = gammadata_.gammaUpI_.n_rows();
            // std::cout << "kkold = " << kkold << std::endl;
            upddata_.SetSize(kkold);

            for (size_t i = 0; i < kkold; i++)
            {
                // std::cout << "verticesUpdated_.at(i) = " << verticesUpdated_.at(i) << std::endl;
                upddata_.xup_(i) = greendata_.greenInteractUp_(vertexIndex, verticesUpdated_.at(i));
                upddata_.yup_(i) = greendata_.greenInteractUp_(verticesUpdated_[i], vertexIndex);

                upddata_.xdown_(i) = greendata_.greenInteractDown_(vertexIndex, verticesUpdated_[i]);
                upddata_.ydown_(i) = greendata_.greenInteractDown_(verticesUpdated_[i], vertexIndex);
            }

            MatrixVectorMult(gammadata_.gammaUpI_, upddata_.yup_, 1.0, upddata_.gammaUpIYup_);
            MatrixVectorMult(gammadata_.gammaDownI_, upddata_.ydown_, 1.0, upddata_.gammaDownIYdown_);
            upddata_.dTildeUpI_ = upddata_.dup_ - LinAlg::DotVectors(upddata_.xup_, upddata_.gammaUpIYup_); //this is in fact beta of submatrix gull article (last element of new matrix GammaI)
            upddata_.dTildeDownI_ = upddata_.ddown_ - LinAlg::DotVectors(upddata_.xdown_, upddata_.gammaDownIYdown_);

            ratio = gammakup * gammakdown * upddata_.dTildeUpI_ * upddata_.dTildeDownI_;
        }
        else
        {
            ratio = gammakup * gammakdown * upddata_.dup_ * upddata_.ddown_;
        }
        // std::cout << "after calculate determinant " << std::endl;
        return ratio;
    }

    void AcceptMove(const double &probAcc)
    {
        // std::cout << "in accept move " << std::endl;
        if (probAcc < 0.0)
        {
            dataCT_->sign_ *= -1;
        }

        if (gammadata_.gammaUpI_.n_rows())
        {
            LinAlg::BlockRankOneUpgrade(gammadata_.gammaUpI_, upddata_.gammaUpIYup_, upddata_.xup_, 1.0 / upddata_.dTildeUpI_);
            LinAlg::BlockRankOneUpgrade(gammadata_.gammaDownI_, upddata_.gammaDownIYdown_, upddata_.xdown_, 1.0 / upddata_.dTildeDownI_);
        }
        else
        {
            gammadata_.gammaUpI_ = Matrix_t(1, 1);
            gammadata_.gammaUpI_(0, 0) = 1.0 / upddata_.dup_;
            gammadata_.gammaDownI_ = Matrix_t(1, 1);
            gammadata_.gammaDownI_(0, 0) = 1.0 / upddata_.ddown_;
        }

        // std::cout << "after accept move " << std::endl;
    }

    void
    RemoveVertexSubMatrix()
    {
        // std::cout << "in remove " << std::endl;
        if (vertices0_.size() && verticesRemovable_.size())
        {
            updStats_["Removes"][0]++;
            const size_t ii = static_cast<int>(urng_() * verticesRemovable_.size());
            const size_t vertexIndex = verticesRemovable_.at(ii);

            Vertex vertex = vertices0Tilde_.at(vertexIndex);
            if (vertexIndex >= vertices0_.size())
            {
                RemovePreviouslyInserted(vertexIndex);
            }
            else
            {

                vertex.SetAux(AuxSpin_t::Zero);

                double ratio = CalculateDeterminantRatio(vertex, vertices0Tilde_.at(vertexIndex), vertexIndex);
                double probAcc = double(nPhyscialVertices_) * ratio / KAux();
                probAcc *= PROBINSERT / PROBREMOVE;

                if (urng_() < std::abs(probAcc))
                {
                    updStats_["Removes"][1]++;
                    verticesUpdated_.push_back(vertexIndex);
                    verticesToRemove_.push_back(vertexIndex);
                    verticesRemovable_.erase(verticesRemovable_.begin() + ii);
                    dataCT_->vertices_.at(vertexIndex) = vertex;
                    nPhyscialVertices_ -= 1;
                    nfdata_.FVup_(vertexIndex) = 1.0;
                    nfdata_.FVdown_(vertexIndex) = 1.0;

                    AcceptMove(probAcc);
                }
            }
        }
        // std::cout << "after remove " << std::endl;
    }

    void RemovePreviouslyInserted(const size_t &vertexIndex) //vertexIndex which is in cTilde
    {
        using itType_t = std::vector<size_t>::const_iterator;
        const itType_t ppit = std::find<itType_t, size_t>(verticesUpdated_.begin(), verticesUpdated_.end(), vertexIndex);
        if (ppit == verticesUpdated_.end())
        {
            throw std::runtime_error("Bad index in find vertexIndex in verticesUpdated_!");
        }

        const size_t pp = std::distance<itType_t>(verticesUpdated_.begin(), ppit); //the index of the updated vertex in the gammaSigma Matrices

        const AuxSpin_t auxFrom = dataCT_->vertices_.at(vertexIndex).aux();
        assert(auxFrom != AuxSpin_t::Zero); //we are about to remove a vertex that has been inserted !
        const double gammappupI = -1.0 / gammaUpSubMatrix(auxFrom, AuxSpin_t::Zero);
        const double gammappdownI = -1.0 / gammaDownSubMatrix(auxFrom, AuxSpin_t::Zero);
        const double ratio = gammappupI * gammappdownI * gammadata_.gammaUpI_(pp, pp) * gammadata_.gammaDownI_(pp, pp);
        const double probAcc = PROBINSERT / PROBREMOVE * double(nPhyscialVertices_) * ratio / KAux();

        if (urng_() < std::abs(probAcc))
        {
            nPhyscialVertices_--;
            if (probAcc < 0.0)
            {
                dataCT_->sign_ *= -1;
            }

            updStats_["Removes"][1]++;
            LinAlg::BlockRankOneDowngrade(gammadata_.gammaUpI_, pp);
            LinAlg::BlockRankOneDowngrade(gammadata_.gammaDownI_, pp);

            const size_t kkm1 = verticesUpdated_.size() - 1;
            std::iter_swap(verticesUpdated_.begin() + pp, verticesUpdated_.begin() + kkm1);
            verticesUpdated_.pop_back();

            //Taken from : https://stackoverflow.com/questions/39912/how-do-i-remove-an-item-from-a-stl-vector-with-a-certain-value
            verticesRemovable_.erase(std::remove(verticesRemovable_.begin(), verticesRemovable_.end(), vertexIndex), verticesRemovable_.end());

            //just a check, make sure that the vertex is not in the verticesToRemove just yet
            const itType_t rrit = std::find<itType_t, size_t>(verticesToRemove_.begin(), verticesToRemove_.end(), vertexIndex);
            if (rrit != verticesToRemove_.end())
            {
                throw std::runtime_error("Bad index in find vertexIndex in verticesToRemove_!");
            }
            //end of check
            verticesToRemove_.push_back(vertexIndex);

            nfdata_.FVup_(vertexIndex) = 1.0;
            nfdata_.FVdown_(vertexIndex) = 1.0;
            dataCT_->vertices_.at(vertexIndex).SetAux(AuxSpin_t::Zero);
        }
    }

    void InsertVertexSubMatrix()
    {
        if (verticesInsertable_.size())
        {
            updStats_["Inserts"][0]++;
            const size_t ii = static_cast<Site_t>(verticesInsertable_.size() * urng_());
            const size_t vertexIndex = verticesInsertable_.at(ii);
            Vertex vertex = vertices0Tilde_.at(vertexIndex);

            vertex.SetAux(urng_() < 0.5 ? AuxSpin_t::Up : AuxSpin_t::Down);
            double ratio = CalculateDeterminantRatio(vertex, vertices0Tilde_.at(vertexIndex), vertexIndex);
            const size_t kknew = nPhyscialVertices_ + 1;

            double probAcc = KAux() / double(kknew) * ratio;
            probAcc *= PROBREMOVE / PROBINSERT;

            //dont propose to insert the same vertex, even if update rejected.
            verticesInsertable_.erase(verticesInsertable_.begin() + ii);

            if (urng_() < std::abs(probAcc))
            {

                updStats_["Inserts"][1]++;
                verticesUpdated_.push_back(vertexIndex);
                dataCT_->vertices_.at(vertexIndex) = vertex;
                verticesToRemove_.erase(std::remove(verticesToRemove_.begin(), verticesToRemove_.end(), vertexIndex), verticesToRemove_.end());
                verticesRemovable_.push_back(vertexIndex);
                nPhyscialVertices_ += 1;
                nfdata_.FVup_(vertexIndex) = FAuxUp(vertex.aux());
                nfdata_.FVdown_(vertexIndex) = FAuxDown(vertex.aux());

                AcceptMove(probAcc);
            }
        }
    }

    void CleanUpdate()
    {
        //mpiUt::Print("Cleaning, sign, k =  " + std::to_string(dataCT_->sign_) + ",  " + std::to_string(dataCT_->vertices_.size()));
        const size_t kk = dataCT_->vertices_.size();
        if (kk == 0)
        {
            return;
        }
        AssertSizes();
        for (size_t i = 0; i < kk; i++)
        {
            for (size_t j = 0; j < kk; j++)
            {

                nfdata_.Nup_(i, j) = -GetGreenTau0Up(dataCT_->vertices_[i], dataCT_->vertices_[j]) * (nfdata_.FVup_(j) - 1.0);
                nfdata_.Ndown_(i, j) = -GetGreenTau0Down(dataCT_->vertices_[i], dataCT_->vertices_[j]) * (nfdata_.FVdown_(j) - 1.0);

                if (i == j)
                {
                    nfdata_.Nup_(i, i) += nfdata_.FVup_(i);
                    nfdata_.Ndown_(i, i) += nfdata_.FVdown_(i);
                }
            }
        }
        AssertSizes();

        nfdata_.Nup_.Inverse();
        nfdata_.Ndown_.Inverse();
    }

    double GetGreenTau0Up(const Vertex &vertexI, const Vertex &vertexJ) const
    {
        return (dataCT_->green0CachedUp_(vertexI.site(), vertexJ.site(), vertexI.tau() - (vertexJ.tau() + 1e-12)));
    }

    double GetGreenTau0Down(const Vertex &vertexI, const Vertex &vertexJ) const
    {

#ifndef AFM
        return GetGreenTau0Up(vertexI, vertexJ);
#endif

#ifdef AFM
        const double delta = 1e-12;
        // 1e-20;
        Tau_t tauDiff = vertexI.tau() - (vertexJ.tau() + delta);
        Site_t s1 = vertexI.site(); //model_.indepSites().at(ll).first;
        Site_t s2 = vertexJ.site(); //model_.indepSites().at(ll).second;
        return (dataCT_->green0CachedDown_(s1, s2, tauDiff));
#endif
    }

    void Measure()
    {
        AssertSizes();
        SiteVector_t FVupM1 = -(nfdata_.FVup_ - 1.0);
        SiteVector_t FVdownM1 = -(nfdata_.FVdown_ - 1.0);
        DDMGMM(FVupM1, nfdata_.Nup_, *(dataCT_->MupPtr_));
        DDMGMM(FVdownM1, nfdata_.Ndown_, *(dataCT_->MdownPtr_));
        obs_.Measure();
        AssertSizes();
    }

    void SaveMeas()
    {

        obs_.Save();
        mpiUt::SaveConfig(dataCT_->vertices_);
        SaveUpd("upd.meas");
    }

    void SaveTherm()
    {

        SaveUpd("upd.therm");
        for (UpdStats_t::iterator it = updStats_.begin(); it != updStats_.end(); ++it)
        {
            std::string key = it->first;
            updStats_[key] = 0.0;
        }
    }

    void SaveUpd(const std::string fname)
    {
        std::vector<UpdStats_t> updStatsVec;
#ifdef HAVEMPI

        mpi::communicator world;
        if (mpiUt::Rank() == mpiUt::master)
        {
            mpi::gather(world, updStats_, updStatsVec, mpiUt::master);
        }
        else
        {
            mpi::gather(world, updStats_, mpiUt::master);
        }
        if (mpiUt::Rank() == mpiUt::master)
        {
            mpiUt::SaveUpdStats(fname, updStatsVec);
        }

#else
        updStatsVec.push_back(updStats_);
        mpiUt::SaveUpdStats(fname, updStatsVec);
#endif

        mpiUt::Print("Finished Saving MarkovChain.");
    }

    void PreparationSteps()
    {
        // std::cout << "in preparation steps " << std::endl;
        nPhyscialVertices_ = dataCT_->vertices_.size();
        InsertNonInteractVertices();
        EnlargeN();
        UpdateGreenInteract();

        // std::cout << "after preparation steps " << std::endl;
    }

    void UpdateSteps()
    {
        // std::cout << "in update steps " << std::endl;
        UpdateN();
        dataCT_->vertices_.size() > verticesToRemove_.size() ? RemoveNonInteractEfficient() : RemoveNonInteract();
        // std::cout << "after update steps " << std::endl;
    }

    void
    UpdateN()
    {
        // std::cout << "in updatN " << std::endl;
        if (verticesUpdated_.size())
        {
            const size_t NN = vertices0Tilde_.size();
            const size_t LK = verticesUpdated_.size();

            Matrix_t DMatrixUp(NN, NN);
            Matrix_t DMatrixDown(NN, NN);
            DMatrixUp.Zeros();
            DMatrixDown.Zeros();

            Matrix_t GTildeColUp(NN, LK);
            Matrix_t GTildeColDown(NN, LK);
            Matrix_t NTildeRowUp(LK, NN);
            Matrix_t NTildeRowDown(LK, NN);

            for (size_t i = 0; i < NN; i++)
            {
                AuxSpin_t auxI = dataCT_->vertices_.at(i).aux();
                AuxSpin_t auxI0Tilde = vertices0Tilde_.at(i).aux();
                DMatrixUp(i, i) = 1.0 / (1.0 + gammaUpSubMatrix(auxI, auxI0Tilde));
                DMatrixDown(i, i) = 1.0 / (1.0 + gammaDownSubMatrix(auxI, auxI0Tilde));

                for (size_t j = 0; j < verticesUpdated_.size(); j++)
                {
                    GTildeColUp(i, j) = greendata_.greenInteractUp_(i, verticesUpdated_.at(j));
                    GTildeColDown(i, j) = greendata_.greenInteractDown_(i, verticesUpdated_[j]);

                    NTildeRowUp(j, i) = nfdata_.Nup_(verticesUpdated_[j], i);
                    NTildeRowDown(j, i) = nfdata_.Ndown_(verticesUpdated_[j], i);
                }
            }

            Matrix_t tmp(gammadata_.gammaUpI_.n_rows(), NTildeRowUp.n_cols());
            LinAlg::DGEMM(1.0, 0.0, gammadata_.gammaUpI_, NTildeRowUp, tmp);
            LinAlg::DGEMM(-1.0, 1.0, GTildeColUp, tmp, nfdata_.Nup_);

            LinAlg::DGEMM(1.0, 0.0, gammadata_.gammaDownI_, NTildeRowDown, tmp);
            LinAlg::DGEMM(-1.0, 1.0, GTildeColDown, tmp, nfdata_.Ndown_);

            for (size_t j = 0; j < nfdata_.Nup_.n_cols(); j++)
            {
                for (size_t i = 0; i < nfdata_.Nup_.n_rows(); i++)
                {
                    nfdata_.Nup_(i, j) *= DMatrixUp(i, i);
                    nfdata_.Ndown_(i, j) *= DMatrixDown(i, i);
                }
            }

            gammadata_.gammaUpI_.Clear();
            gammadata_.gammaDownI_.Clear();
            // std::cout << "After clear " << std::endl;
        }

        // std::cout << "After updateN " << std::endl;
    }

    void EnlargeN()
    {
        // std::cout << "in EnlargeN " << std::endl;
        //build the B matrices
        const size_t N0 = vertices0_.size(); //here vertices0 is the same as vertices withouth the non-interacting spins
        if (N0)
        {
            Matrix_t BUp(KMAX_UPD_, N0);
            Matrix_t BDown(KMAX_UPD_, N0);

            for (size_t j = 0; j < N0; j++)
            {
                for (size_t i = 0; i < KMAX_UPD_; i++)
                {
                    Vertex vertexI = vertices0Tilde_.at(N0 + i);
                    Vertex vertexJ = dataCT_->vertices_.at(j);
                    BUp(i, j) = GetGreenTau0Up(vertexI, vertexJ) * (nfdata_.FVup_(j) - 1.0);
                    BDown(i, j) = GetGreenTau0Up(vertexI, vertexJ) * (nfdata_.FVdown_(j) - 1.0);
                }
            }

            Matrix_t BUpMTilde(BUp.n_rows(), nfdata_.Nup_.n_cols());
            Matrix_t BDownMTilde(BDown.n_rows(), nfdata_.Ndown_.n_cols());
            LinAlg::DGEMM(1.0, 0.0, BUp, nfdata_.Nup_, BUpMTilde);
            LinAlg::DGEMM(1.0, 0.0, BDown, nfdata_.Ndown_, BDownMTilde);

            const size_t newSize = N0 + KMAX_UPD_;

            nfdata_.Nup_.Resize(newSize, newSize);
            nfdata_.Ndown_.Resize(newSize, newSize);

            nfdata_.FVup_.resize(newSize);
            (nfdata_.FVdown_).resize(newSize);

            //utiliser Lapack ici ?
            for (size_t i = N0; i < newSize; i++)
            {
                nfdata_.FVup_(i) = 1.0;
                (nfdata_.FVdown_)(i) = 1.0;
            }

            //utiliser Lapack ici, slacpy, ?
            Matrix_t eye(newSize - N0, newSize - N0);
            eye.Eye();
            nfdata_.Nup_.SubMat(0, N0, N0 - 1, newSize - 1, 0.0);
            nfdata_.Nup_.SubMat(N0, 0, newSize - 1, N0 - 1, BUpMTilde);
            nfdata_.Nup_.SubMat(N0, N0, newSize - 1, newSize - 1, eye);

            nfdata_.Ndown_.SubMat(0, N0, N0 - 1, newSize - 1, 0.0);
            nfdata_.Ndown_.SubMat(N0, 0, newSize - 1, N0 - 1, BDownMTilde);
            nfdata_.Ndown_.SubMat(N0, N0, newSize - 1, newSize - 1, eye);
        }
        else
        {

            nfdata_.Nup_ = Matrix_t(KMAX_UPD_, KMAX_UPD_);
            nfdata_.Nup_.Eye();
            nfdata_.Ndown_ = Matrix_t(KMAX_UPD_, KMAX_UPD_);
            nfdata_.Ndown_.Eye();
            nfdata_.FVup_ = SiteVector_t(KMAX_UPD_).ones();
            nfdata_.FVdown_ = SiteVector_t(KMAX_UPD_).ones();
        }
        // std::cout << "after EnlargeN " << std::endl;
    }

    void UpdateGreenInteract()
    {

        // std::cout << "in greeninteract " << std::endl;
        const size_t N0 = vertices0_.size();
        const size_t NN = N0 + KMAX_UPD_;
        //assert(NN == vertices0Tilde_.size());
        Matrix_t green0up(NN, KMAX_UPD_);

        //ici updater que ce qui est necessaire, updater seulement les colonnes updater
        greendata_.greenInteractUp_ = nfdata_.Nup_;
        greendata_.greenInteractDown_ = nfdata_.Ndown_;
        for (size_t j = 0; j < N0; j++)
        {

            double factup = 1.0 / (nfdata_.FVup_(j) - 1.0);
            greendata_.greenInteractUp_.MultCol(j, nfdata_.FVup_(j) * factup);
            greendata_.greenInteractUp_(j, j) -= factup;

            double factdown = 1.0 / (nfdata_.FVdown_(j) - 1.0);
            greendata_.greenInteractDown_.MultCol(j, nfdata_.FVdown_(j) * factdown);
            greendata_.greenInteractDown_(j, j) -= factdown;
        }

        for (size_t j = 0; j < KMAX_UPD_; j++)
        {

            for (size_t i = 0; i < NN; i++)
            {
                green0up(i, j) = GetGreenTau0Up(vertices0Tilde_.at(i), vertices0Tilde_.at(N0 + j));
            }
        }

        LinAlg::DGEMM(1.0, 0.0, nfdata_.Nup_, green0up, greendata_.greenInteractUp_, N0);
        LinAlg::DGEMM(1.0, 0.0, nfdata_.Ndown_, green0up, greendata_.greenInteractDown_, N0);

        // std::cout << "after greeninteract " << std::endl;
    }

    void RemoveNonInteract()
    {
        // std::cout << "in removenoninteract " << std::endl;
        std::sort(verticesToRemove_.begin(), verticesToRemove_.end());

        for (size_t i = 0; i < verticesToRemove_.size(); i++)
        {
            size_t index = verticesToRemove_[i] - i; //if verticesToRemove_ in increasing order of index

            //cela serait plus rapide de faire un swap et ensuite d'enlever les dernieres colones?
            nfdata_.Nup_.ShedRowAndCol(index);

            nfdata_.Ndown_.ShedRowAndCol(index);

            nfdata_.FVup_.shed_row(index);
            (nfdata_.FVdown_).shed_row(index);

            dataCT_->vertices_.erase(dataCT_->vertices_.begin() + index);
        }

        verticesToRemove_.clear();
        verticesInsertable_.clear();
        verticesUpdated_.clear();

        //assert(nfdata_.Ndown_.n_rows == dataCT_->vertices_.size());
        verticesRemovable_.clear();
        AssertSanity();
        // std::cout << "after remvovenoninteract " << std::endl;
    }

    void RemoveNonInteractEfficient()
    {
        // std::cout << "in removenoninteractEfficient " << std::endl;
        std::sort(verticesToRemove_.begin(), verticesToRemove_.end());
        std::vector<size_t> verticesInteracting;
        // std::cout << "vertices.size() = " << dataCT_->vertices_.size() << std::endl;
        // std::cout << "verticesToRemove.size() = " << verticesToRemove_.size() << std::endl;
        for (size_t ii = 0; ii < vertices0Tilde_.size(); ii++)
        {
            verticesInteracting.push_back(ii);
        }

        // std::cout << "Here 1" << std::endl;
        for (size_t ii = 0; ii < verticesToRemove_.size(); ii++)
        {
            // std::cout << "in loop  " << ii << std::endl;
            size_t indexToRemove = verticesToRemove_.at(ii) - ii;
            verticesInteracting.erase(verticesInteracting.begin() + indexToRemove);
        }
        const size_t INTERS = verticesInteracting.size(); //interact size
        // assert(INTERS > verticesToRemove_.size());
        // std::cout << "Here 2" << std::endl;

        for (size_t i = 0; i < std::min<size_t>(verticesInteracting.size(), verticesToRemove_.size()); i++)
        {
            size_t indexToRemove = verticesToRemove_.at(i);
            size_t indexInteracting = verticesInteracting.at(INTERS - 1 - i);
            if (indexInteracting > indexToRemove)
            {
                nfdata_.Nup_.SwapRowsAndCols(indexToRemove, indexInteracting);
                nfdata_.Ndown_.SwapRowsAndCols(indexToRemove, indexInteracting);
                // std::cout << "in if " << std::endl;
                nfdata_.FVup_.swap_rows(indexToRemove, indexInteracting);
                nfdata_.FVdown_.swap_rows(indexToRemove, indexInteracting);

                std::iter_swap(dataCT_->vertices_.begin() + indexToRemove, dataCT_->vertices_.begin() + indexInteracting);
            }
            // std::cout << "After if " << std::endl;
        }

        nfdata_.Nup_.Resize(INTERS, INTERS);
        nfdata_.Ndown_.Resize(INTERS, INTERS);

        nfdata_.FVup_.resize(INTERS);
        nfdata_.FVdown_.resize(INTERS);
        dataCT_->vertices_.resize(INTERS);

        verticesToRemove_.clear();
        verticesInsertable_.clear();
        verticesUpdated_.clear();

        assert(nfdata_.Ndown_.n_rows() == dataCT_->vertices_.size());
        verticesRemovable_.clear();
        AssertSanity();
        // std::cout << "after remvovenoninteractEfficient " << std::endl;
    }

    void InsertNonInteractVertices()
    {
        //AssertSanity();
        vertices0_ = dataCT_->vertices_;
        const size_t N0 = dataCT_->vertices_.size();
        assert(N0 == nfdata_.Nup_.n_rows());

        for (size_t i = 0; i < KMAX_UPD_; i++)
        {
            Vertex vertex(dataCT_->beta_ * urng_(), static_cast<Site_t>(Nc * urng_()), AuxSpin_t::Zero);
            dataCT_->vertices_.push_back(vertex);

            verticesInsertable_.push_back(N0 + i);
        }
        verticesToRemove_ = verticesInsertable_;

        for (size_t i = 0; i < N0; i++)
        {
            verticesRemovable_.push_back(i);
        }

        vertices0Tilde_ = dataCT_->vertices_;
    }

    void AssertSanity()
    {
        assert(gammadata_.gammaDownI_.n_rows() == gammadata_.gammaUpI_.n_cols());

        for (size_t i = 0; i < dataCT_->vertices_.size(); i++)
        {
            // std::cout << "i = " << i << std::endl;
            AuxSpin_t aux = dataCT_->vertices_.at(i).aux();
            // std::cout << "FAuxUp(aux),  nfdata_.FVup_(i)= " << FAuxUp(aux) << ", " << nfdata_.FVup_(i) << std::endl;
            assert(aux != AuxSpin_t::Zero);
            assert(std::abs(FAuxUp(aux) - nfdata_.FVup_(i)) < 1e-10);
            assert(std::abs(FAuxDown(aux) - nfdata_.FVdown_(i)) < 1e-10);
        }
    }

    void PrintVector(std::vector<size_t> v)
    {
        for (size_t i = 0; i < v.size(); i++)
        {
            std::cout << v.at(i) << " ";
        }
        std::cout << std::endl;
    }

  protected:
    //attributes
    TModel model_;
    Utilities::EngineTypeMt19937_t rng_;
    Utilities::UniformRngMt19937_t urng_;
    GammaData gammadata_;
    UpdData upddata_;
    NFData nfdata_;
    GreenData greendata_;

    std::shared_ptr<Obs::ISDataCT<TIOModel, TModel>> dataCT_;
    Obs::Observables<TIOModel, TModel> obs_;

    std::vector<size_t> verticesUpdated_;
    std::vector<size_t> verticesRemovable_; //the interacting vertices that can be removed
    std::vector<size_t> verticesToRemove_;
    std::vector<size_t> verticesInsertable_; //the vertices at which you can insert
    std::vector<Vertex> vertices0_;          //the initial config, withouth the noninteracting vertices
    std::vector<Vertex> vertices0Tilde_;

    UpdStats_t updStats_; //[0] = number of propsed, [1]=number of accepted

    const size_t KMAX_UPD_;

    const double expUp_;
    const double expDown_;

    size_t nPhyscialVertices_;
    size_t updatesProposed_;
};

template <typename TIOModel, typename TModel>
ABC_MarkovChainSubMatrix<TIOModel, TModel>::~ABC_MarkovChainSubMatrix() {} //destructors must exist
} // namespace Markov

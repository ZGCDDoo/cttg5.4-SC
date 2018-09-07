#pragma once
#include <valarray>

#include "../Utilities/Utilities.hpp"
#include "../Utilities/LinAlg.hpp"
#include "../Utilities/Matrix.hpp"
#include "../Utilities/MPIUtilities.hpp"
#include "../Utilities/Fourier.hpp"
#include "../Utilities/GreenTau.hpp"
#include "Obs/Observables.hpp"
#include "ISData.hpp"

//#define DEBUG_TEST

namespace Markov
{

using Fourier::MatToTau;
using Fourier::MatToTauCluster;
using Vertex = Utilities::Vertex;
typedef LinAlg::Matrix_t Matrix_t;

struct NFData
{

    NFData() : F_(), N_(), dummy_(){};
    SiteVector_t F_;
    Matrix_t N_;
    Matrix_t dummy_;
};

template <typename TIOModel, typename TModel>
class ABC_MarkovChain
{

    using GreenTau_t = GreenTau::GreenCluster0Tau<TIOModel>;

  public:
    const size_t Nc = TModel::Nc;
    // const double PROBFLIP = 0.25;
    const double PROBINSERT = 0.25;
    const double PROBREMOVE = 1.0 - PROBINSERT;

    ABC_MarkovChain(const Json &jj, const size_t &seed) : modelPtr_(new TModel(jj)),
                                                          rng_(seed),
                                                          urng_(rng_, Utilities::UniformDistribution_t(0.0, 1.0)),
                                                          nfdata_(),
                                                          dataCT_(
                                                              new Obs::ISDataCT<TIOModel, TModel>(
                                                                  jj["beta"].get<double>(),
                                                                  *modelPtr_, jj["NTAU"].get<double>())),
                                                          obs_(dataCT_, jj),
                                                          expUp_(std::exp(modelPtr_->gamma())),
                                                          expDown_(std::exp(-modelPtr_->gamma()))
    {
        const std::valarray<size_t> zeroPair = {0, 0};
        updStats_["Inserts"] = zeroPair;
        updStats_["Removes"] = zeroPair;
        updStats_["Flips"] = zeroPair;
        updatesProposed_ = 0;

        mpiUt::Print("MarkovChain Created \n");
    }

    virtual ~ABC_MarkovChain() = 0;

    //Getters
    TModel model() const
    {
        return (*modelPtr_);
    };

    Matrix_t N() const
    {
        return nfdata_.N_;
    };

    std::vector<Vertex> vertices() const
    {
        return dataCT_->vertices_;
    };

    size_t updatesProposed() const { return updatesProposed_; }

    double beta() const
    {
        return dataCT_->beta_;
    };

    virtual double gammaUpTrad(const AuxSpin_t &auxxTo, const AuxSpin_t &vauxFrom) = 0;
    virtual double gammaDownTrad(const AuxSpin_t &auxxTo, const AuxSpin_t &vauxFrom) = 0;
    virtual double KAux() = 0;
    virtual double FAuxUp(const AuxSpin_t &aux) = 0;
    virtual double FAuxDown(const AuxSpin_t &aux) = 0;

    void ThermalizeFromConfig()
    {
        assert(false);
    }

    void DoStep()
    {

        urng_() < PROBINSERT ? InsertVertex() : RemoveVertex();

        updatesProposed_++;
    }

    void AssertSizes()
    {
        const size_t kk = dataCT_->vertices_.size();
        assert(2 * kk == nfdata_.N_.n_rows());
        assert(2 * kk == nfdata_.N_.n_cols());
        assert(2 * kk == nfdata_.F_.n_elem());
    }

    void InsertVertex()
    {
        //AssertSizes();
        updStats_["Inserts"][0]++;
        Vertex vertex = Vertex(dataCT_->beta_ * urng_(), static_cast<Site_t>(Nc * urng_()), urng_() < 0.5 ? AuxSpin_t::Up : AuxSpin_t::Down);
        const double fauxup = FAuxUp(vertex.aux());
        const double fauxdown = FAuxDown(vertex.aux());
        const double fauxupM1 = fauxup - 1.0;
        const double fauxdownM1 = fauxdown - 1.0;

        const double sUp = fauxup - GetGreenTau0Up(vertex, vertex) * fauxupM1;
        const double sDown = fauxdown - GetGreenTau0Down(vertex, vertex) * fauxdownM1;

        if (dataCT_->vertices_.size())
        {
            //AssertSizes();
            const size_t kkold = dataCT_->vertices_.size();
            const size_t kknew = kkold + 2;

            SiteVector_t newLastColUp_(kkold);
            SiteVector_t newLastRowUp_(kkold);
            SiteVector_t newLastColDown_(kkold);
            SiteVector_t newLastRowDown_(kkold);

            double sTildeUpI = sUp;
            double sTildeDownI = sDown;

            //Probably put this in a method
            for (size_t i = 0; i < kkold; i++)
            {
                newLastRowUp_(i) = -GetGreenTau0Up(vertex, dataCT_->vertices_.at(i)) * (nfdata_.FVup_(i) - 1.0);
                newLastColUp_(i) = -GetGreenTau0Up(dataCT_->vertices_[i], vertex) * fauxupM1;

                newLastRowDown_(i) = -GetGreenTau0Down(vertex, dataCT_->vertices_[i]) * (nfdata_.FVdown_(i) - 1.0);
                newLastColDown_(i) = -GetGreenTau0Down(dataCT_->vertices_[i], vertex) * fauxdownM1;
            }

            SiteVector_t NQUp(kkold); //NQ = N*Q
            SiteVector_t NQDown(kkold);
            MatrixVectorMult(nfdata_.Nup_, newLastColUp_, 1.0, NQUp);
            MatrixVectorMult(nfdata_.Ndown_, newLastColDown_, 1.0, NQDown);
            sTildeUpI -= LinAlg::DotVectors(newLastRowUp_, NQUp);
            sTildeDownI -= LinAlg::DotVectors(newLastRowDown_, NQDown);

            const double ratio = sTildeUpI * sTildeDownI;
            const double ratioAcc = PROBREMOVE / PROBINSERT * KAux() / kknew * ratio;
            //AssertSizes();
            if (urng_() < std::abs(ratioAcc))
            {
                updStats_["Inserts"][1]++;
                if (probAcc < .0)
                {
                    dataCT_->sign_ *= -1;
                }

                LinAlg::BlockRankOneTwoUpgrade(nfdata_.Nup_, NQUp, newLastRowUp_, 1.0 / sTildeUpI);
                LinAlg::BlockRankOneTwoUpgrade(nfdata_.Ndown_, NQDown, newLastRowDown_, 1.0 / sTildeDownI);
                nfdata_.F_.resize(kknew);
                nfdata_.F_(kkold) = fauxup;
                nfdata_.F_(kkold + 1) = fauxdown;
                dataCT_->vertices_.push_back(vertex);
                //AssertSizes();
            }
        }
        else
        {
            //AssertSizes();
            const double ratioAcc = PROBREMOVE / PROBINSERT * KAux() * sUp * sDown;
            if (urng_() < std::abs(ratioAcc))
            {
                if (ratioAcc < .0)
                {
                    dataCT_->sign_ *= -1;
                }

                nfdata_.N_ = {{1.0 / sUp, 0.0},
                              {0.0, 1.0 / sDown}};

                nfdata_.F_ = SiteVector_t(2);
                nfdata_.F_(0) = fauxup;
                nfdata_.F_(1) = fauxdown;

                dataCT_->vertices_.push_back(vertex);
            }
            //AssertSizes();
        }

        return;
    }

    void
    RemoveVertex()
    {
        //AssertSizes();
        updStats_["Removes"][0]++;
        if (dataCT_->vertices_.size())
        {
            const size_t pp = static_cast<int>(urng_() * dataCT_->vertices_.size());

            double probAcc = static_cast<double>(dataCT_->vertices_.size()) / KAux() * nfdata_.Nup_(pp, pp) * nfdata_.Ndown_(pp, pp);
            probAcc *= PROBINSERT / PROBREMOVE;

            if (urng_() < std::abs(probAcc))
            {
                //AssertSizes();
                updStats_["Removes"][1]++;
                if (probAcc < .0)
                {
                    dataCT_->sign_ *= -1; //not to sure here, should it not just be sign = -1 ??
                }

                //The update matrices of size k-1 x k-1 with the pp row and col deleted and the last row and col now at index pp

                const size_t kk = dataCT_->vertices_.size();
                const size_t kkm1 = kk - 1;

                LinAlg::BlockRankOneDowngrade(nfdata_.Nup_, pp);
                LinAlg::BlockRankOneDowngrade(nfdata_.Ndown_, pp);

                nfdata_.FVup_.swap_rows(pp, kkm1);
                nfdata_.FVdown_.swap_rows(pp, kkm1);
                nfdata_.FVup_.resize(kkm1);
                nfdata_.FVdown_.resize(kkm1);

                std::iter_swap(dataCT_->vertices_.begin() + pp, dataCT_->vertices_.begin() + kkm1); //swap the last vertex and the vertex pp in vertices.
                                                                                                    //to be consistent with the updated Mup and dataCT_->Mdown_
                dataCT_->vertices_.pop_back();
                //AssertSizes();
            }
        }
    }

    void CleanUpdate(bool print = false)
    {
        //mpiUt::Print("Cleaning, sign, k =  " + std::to_string(dataCT_->sign_) + ",  " + std::to_string(dataCT_->vertices_.size()));
        const size_t kk = dataCT_->vertices_.size();
        if (kk == 0)
        {
            return;
        }

        //AssertSizes();
        for (size_t i = 0; i < kk; i++)
        {
            for (size_t j = 0; j < kk; j++)
            {

                nfdata_.Nup_(i, j) = -GetGreenTau0Up(dataCT_->vertices_.at(i), dataCT_->vertices_.at(j)) * (nfdata_.FVup_(j) - 1.0);
                nfdata_.Ndown_(i, j) = -GetGreenTau0Down(dataCT_->vertices_[i], dataCT_->vertices_[j]) * (nfdata_.FVdown_(j) - 1.0);

                if (i == j)
                {
                    nfdata_.Nup_(i, i) += nfdata_.FVup_(i);
                    nfdata_.Ndown_(i, i) += nfdata_.FVdown_(i);
                }
            }
        }
        //AssertSizes();
        if (print)
        {
            SiteVector_t FVupM1 = -(nfdata_.FVup_ - 1.0);
            SiteVector_t FVdownM1 = -(nfdata_.FVdown_ - 1.0);
            DDMGMM(FVupM1, nfdata_.Nup_, *(dataCT_->MupPtr_));
            DDMGMM(FVdownM1, nfdata_.Ndown_, *(dataCT_->MdownPtr_));
            (*(dataCT_->MupPtr_)).Print();
            (*(dataCT_->MdownPtr_)).Print();
        }
        nfdata_.Nup_.Inverse();
        nfdata_.Ndown_.Inverse();
    }

    double GetGreenTau0Up(const Vertex &vertexI, const Vertex &vertexJ) const
    {
        return (dataCT_->green0CachedUp_(vertexI.site(), vertexJ.site(), vertexI.tau() - vertexJ.tau()));
    }

    // In fact return -g_Down(-tau) (nambu version of gdown)
    double GetGreenTau0Down(const Vertex &vertexI, const Vertex &vertexJ) const
    {

#ifdef AFM
        return (-dataCT_->green0CachedDown_(vertexI.site(), vertexJ.site(), -(vertexI.tau() - vertexJ.tau())));
#else
        return (-dataCT_->green0CachedUp_(vertexI.site(), vertexJ.site(), -(vertexI.tau() - vertexJ.tau())));

#endif
    }

    void Measure()
    {
        SiteVector_t FVupM1 = -(nfdata_.FVup_ - 1.0);
        SiteVector_t FVdownM1 = -(nfdata_.FVdown_ - 1.0);
        DDMGMM(FVupM1, nfdata_.Nup_, *(dataCT_->MupPtr_));
        DDMGMM(FVdownM1, nfdata_.Ndown_, *(dataCT_->MdownPtr_));
        obs_.Measure();
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

  protected:
    //attributes
    std::shared_ptr<TModel> modelPtr_;
    Utilities::EngineTypeMt19937_t rng_;
    Utilities::UniformRngMt19937_t urng_;
    NFData nfdata_;
    std::shared_ptr<Obs::ISDataCT<TIOModel, TModel>> dataCT_;
    Obs::Observables<TIOModel, TModel> obs_;

    UpdStats_t updStats_; //[0] = number of propsed, [1]=number of accepted

    const double expUp_;
    const double expDown_;
    size_t updatesProposed_;
};

template <typename TIOModel, typename TModel>
ABC_MarkovChain<TIOModel, TModel>::~ABC_MarkovChain() {} //destructors must exist

} // namespace Markov

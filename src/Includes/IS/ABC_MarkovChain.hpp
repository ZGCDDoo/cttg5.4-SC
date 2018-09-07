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

    NFData() : FVup_(), FVdown_(), Nup_(), Ndown_(), dummy_(){};
    SiteVector_t FVup_;
    SiteVector_t FVdown_;
    Matrix_t Nup_;
    Matrix_t Ndown_;
    Matrix_t dummy_;
};

template <typename TIOModel, typename TModel>
class ABC_MarkovChain
{

    using GreenTau_t = GreenTau::GreenCluster0Tau<TIOModel>;

  public:
    const size_t Nc = TModel::Nc;
    const double PROBFLIP = 0.25;
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
        if (mpiUt::LoadConfig(dataCT_->vertices_))
        {
            const size_t kk = dataCT_->vertices_.size();
            nfdata_.FVup_ = SiteVector_t(kk);
            nfdata_.FVdown_ = SiteVector_t(kk);
            for (size_t i = 0; i < kk; i++)
            {
                AuxSpin_t aux = dataCT_->vertices_.at(i).aux();
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

        if (urng_() < PROBFLIP)
        {
            FlipAux();
        }
        else
        {
            urng_() < PROBINSERT ? InsertVertex() : RemoveVertex();
        }

        updatesProposed_++;
    }

    void FlipAux()
    {

        if (dataCT_->vertices_.size())
        {
            updStats_["Flips"][0]++;
            size_t p = static_cast<size_t>(dataCT_->vertices_.size() * urng_());
            Vertex vertex = dataCT_->vertices_.at(p);
            vertex.FlipAux();
            AuxSpin_t auxTo = vertex.aux();
            AuxSpin_t auxFrom = dataCT_->vertices_.at(p).aux();

            double fauxup = nfdata_.FVup_(p);
            double fauxdown = nfdata_.FVdown_(p);
            double fauxupM1 = fauxup - 1.0;
            double fauxdownM1 = fauxdown - 1.0;
            double gammakup = gammaUpTrad(auxTo, auxFrom);
            double gammakdown = gammaDownTrad(auxTo, auxFrom);

            double ratioUp = 1.0 + (1.0 - (nfdata_.Nup_(p, p) * fauxup - 1.0) / (fauxupM1)) * gammakup;
            double ratioDown = 1.0 + (1.0 - (nfdata_.Ndown_(p, p) * fauxdown - 1.0) / (fauxdownM1)) * gammakdown;

            double probAcc = ratioUp * ratioDown;

            if (urng_() < std::abs(probAcc))
            {
                updStats_["Flips"][1]++;
                if (probAcc < 0.0)
                {
                    dataCT_->sign_ *= -1;
                }

                //AssertSizes();
                const size_t kk = dataCT_->vertices_.size();
                double lambdaUp = gammakup / ratioUp;
                double lambdaDown = gammakdown / ratioDown;

                SiteVector_t rowpUp;
                SiteVector_t colpUp;
                LinAlg::ExtractRow(p, rowpUp, nfdata_.Nup_);
                LinAlg::ExtractCol(p, colpUp, nfdata_.Nup_);

                SiteVector_t rowpDown;
                SiteVector_t colpDown;
                LinAlg::ExtractRow(p, rowpDown, nfdata_.Ndown_);
                LinAlg::ExtractCol(p, colpDown, nfdata_.Ndown_);

                for (size_t j = 0; j < kk; j++)
                {
                    for (size_t i = 0; i < kk; i++)
                    {
                        if (i != p)
                        {
                            nfdata_.Nup_(i, j) += (colpUp(i) * fauxup / fauxupM1) * lambdaUp * rowpUp(j);
                            nfdata_.Ndown_(i, j) += (colpDown(i) * fauxdown / fauxdownM1) * lambdaDown * rowpDown(j);
                        }
                        else
                        {
                            nfdata_.Nup_(i, j) += (((colpUp(i) * fauxup - 1.0) / fauxupM1) - 1.0) * lambdaUp * rowpUp(j);
                            nfdata_.Ndown_(i, j) += (((colpDown(i) * fauxdown - 1.0) / fauxdownM1) - 1.0) * lambdaDown * rowpDown(j);
                        }
                    }
                }

                dataCT_->vertices_.at(p) = vertex;
                nfdata_.FVup_(p) = fauxdown;
                nfdata_.FVdown_(p) = fauxup;

                //AssertSizes();
            }
        }
    }

    void AssertSizes()
    {
        const size_t kk = dataCT_->vertices_.size();
        assert(kk == nfdata_.Nup_.n_rows());
        assert(kk == nfdata_.Nup_.n_cols());
        assert(kk == nfdata_.Ndown_.n_rows());
        assert(kk == nfdata_.Ndown_.n_cols());
    }

    void InsertVertex()
    {
        //AssertSizes();
        updStats_["Inserts"][0]++;
        Vertex vertex = Vertex(dataCT_->beta_ * urng_(), static_cast<Site_t>(Nc * urng_()), urng_() < 0.5 ? AuxSpin_t::Up : AuxSpin_t::Down);
        double fauxup = FAuxUp(vertex.aux());
        double fauxdown = FAuxDown(vertex.aux());
        double fauxupM1 = fauxup - 1.0;
        double fauxdownM1 = fauxdown - 1.0;

        double sUp = fauxup - GetGreenTau0Up(vertex, vertex) * fauxupM1;
        double sDown = fauxdown - GetGreenTau0Down(vertex, vertex) * fauxdownM1;

        if (dataCT_->vertices_.size())
        {
            //AssertSizes();
            const size_t kkold = dataCT_->vertices_.size();
            const size_t kknew = kkold + 1;

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
            double probAcc = KAux() / kknew * ratio;
            probAcc *= PROBREMOVE / PROBINSERT;
            //AssertSizes();
            if (urng_() < std::abs(probAcc))
            {
                updStats_["Inserts"][1]++;
                if (probAcc < .0)
                {
                    dataCT_->sign_ *= -1;
                }

                LinAlg::BlockRankOneUpgrade(nfdata_.Nup_, NQUp, newLastRowUp_, 1.0 / sTildeUpI);
                LinAlg::BlockRankOneUpgrade(nfdata_.Ndown_, NQDown, newLastRowDown_, 1.0 / sTildeDownI);
                nfdata_.FVup_.resize(kknew);
                nfdata_.FVdown_.resize(kknew);
                nfdata_.FVup_(kkold) = fauxup;
                nfdata_.FVdown_(kkold) = fauxdown;
                dataCT_->vertices_.push_back(vertex);
                //AssertSizes();
            }
        }
        else
        {
            //AssertSizes();
            double probAcc = KAux() * sUp * sDown;
            probAcc *= PROBREMOVE / PROBINSERT;
            if (urng_() < std::abs(probAcc))
            {
                if (probAcc < .0)
                {
                    dataCT_->sign_ *= -1;
                }

                nfdata_.Nup_ = Matrix_t(1, 1);
                nfdata_.Ndown_ = Matrix_t(1, 1);
                nfdata_.Nup_(0, 0) = 1.0 / sUp;
                nfdata_.Ndown_(0, 0) = 1.0 / sDown;

                nfdata_.FVup_ = SiteVector_t(1);
                nfdata_.FVdown_ = SiteVector_t(1);
                nfdata_.FVup_(0) = fauxup;
                nfdata_.FVdown_(0) = fauxdown;

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

    double GetGreenTau0Down(const Vertex &vertexI, const Vertex &vertexJ) const
    {

#ifdef AFM
        return (dataCT_->green0CachedDown_(vertexI.site(), vertexJ.site(), vertexI.tau() - vertexJ.tau()));
#else
        return GetGreenTau0Up(vertexI, vertexJ);

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

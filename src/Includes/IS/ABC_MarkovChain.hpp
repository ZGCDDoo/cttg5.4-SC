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
        assert(2 * kk == nfdata_.F_.n_elem);
        // std::cout << "kk = " << kk << std::endl;
    }

    void InsertVertex()
    {
        // std::cout << "In insertvertex" << std::endl;

        AssertSizes();
        updStats_["Inserts"][0]++;
        Vertex vertex = Vertex(dataCT_->beta_ * urng_(), static_cast<Site_t>(Nc * urng_()), urng_() < 0.5 ? AuxSpin_t::Up : AuxSpin_t::Down);
        const double fauxup = FAuxUp(vertex.aux());
        const double fauxdown = FAuxDown(vertex.aux());
        const double fauxupM1 = fauxup - 1.0;
        const double fauxdownM1 = fauxdown - 1.0;

        const double sUp = fauxup - GetGreenTau0Up(vertex, vertex) * fauxupM1; // The
        const double sDown = fauxdown - GetGreenTau0Down(vertex, vertex) * fauxdownM1;

        if (dataCT_->vertices_.size())
        {
            AssertSizes();
            const size_t kkold = dataCT_->vertices_.size();
            const size_t kknew = kkold + 1;

            Matrix_t Q_(2 * kkold, 2);
            Matrix_t R_(2, 2 * kkold);
            //Probably put this in a method

            // std::cout << "In INsertvertex before loop " << std::endl;
            for (size_t i = 0; i < kkold; i++)
            {

                //consider the vertices being numbered from one to L, then, for the jth vertex (we are adding the pth vertex):
                const Vertex vertexI = dataCT_->vertices_.at(i);
                Q_(2 * i, 0) = -GetGreenTau0Up(vertexI, vertex) * fauxupM1;         // G^{Up, Up}_{j, p}
                Q_(2 * i, 1) = -GetFTau0UpDown(vertexI, vertex);                    // F_{Up Down}_{j, p}
                Q_(2 * i + 1, 0) = -GetFTau0UpDown(vertexI, vertex);                // F_{Down, Up}_{j, p}
                Q_(2 * i + 1, 1) = -GetGreenTau0Down(vertexI, vertex) * fauxdownM1; // G_{Down, Down}_{j, p}

                R_(0, 2 * i) = -GetGreenTau0Up(vertex, vertexI) * (nfdata_.F_(2 * i) - 1.0);           // G^{Up, Up}_{p, j}
                R_(0, 2 * i + 1) = -GetFTau0UpDown(vertex, vertexI);                                   // F_{Up Down}_{j, p}
                R_(1, 2 * i) = -GetFTau0DownUp(vertex, vertexI);                                       // F_{Down, Up}_{j, p}
                R_(1, 2 * i + 1) = -GetGreenTau0Down(vertex, vertexI) * (nfdata_.F_(2 * i + 1) - 1.0); // G_{Down, Down}_{j, p}
            }
            // std::cout << "In INsertvertex After loop " << std::endl;

            //Watch out, we are calculating two times the matrix NQ, once here and once in ranktwoupgrade. In a next version, only calculate here, not in ranktwoupgrade.
            // Matrix_t NQ(2 * kkold, 2); //NQ = N*Q
            // MatrixVectorMult(nfdata_.N_, Q_, 1.0, NQUp);

            // Matrix_t RNQ(2, 2); //R*NQ
            Matrix_t sTilde = Matrix_t({{sUp, 0.0}, {0.0, sDown}}) - LinAlg::DotRank2(R_, nfdata_.N_, Q_);
            sTilde.Inverse();
            const double ratioAcc = PROBREMOVE / PROBINSERT * KAux() / kknew * 1.0 / sTilde.Determinant();

            AssertSizes();
            if (urng_() < std::abs(ratioAcc))
            {
                updStats_["Inserts"][1]++;
                if (ratioAcc < .0)
                {
                    dataCT_->sign_ *= -1;
                }

                LinAlg::BlockRankTwoUpgrade(nfdata_.N_, Q_, R_, sTilde);
                nfdata_.F_.resize(2 * kkold + 2);
                nfdata_.F_(2 * kkold) = fauxup;
                nfdata_.F_(2 * kkold + 1) = fauxdown;
                dataCT_->vertices_.push_back(vertex);
                AssertSizes();
            }
        }
        else
        {
            AssertSizes();
            const double ratioAcc = PROBREMOVE / PROBINSERT * KAux() * sUp * sDown;
            if (urng_() < std::abs(ratioAcc))
            {
                if (ratioAcc < 0.0)
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
            AssertSizes();
        }

        // std::cout << "After insertvertex" << std::endl;
    }

    void RemoveVertex()
    {
        AssertSizes();
        updStats_["Removes"][0]++;
        if (dataCT_->vertices_.size())
        {
            const size_t pp = static_cast<int>(urng_() * dataCT_->vertices_.size());

            const ClusterMatrix_t STildeInverse = {{nfdata_.N_(2 * pp, 2 * pp), nfdata_.N_(2 * pp, 2 * pp + 1)}, {nfdata_.N_(2 * pp + 1, 2 * pp), nfdata_.N_(2 * pp + 1, 2 * pp + 1)}};
            const double ratioAcc = PROBINSERT / PROBREMOVE * static_cast<double>(dataCT_->vertices_.size()) / KAux() * arma::det(STildeInverse);

            if (urng_() < std::abs(ratioAcc))
            {
                AssertSizes();
                updStats_["Removes"][1]++;
                if (ratioAcc < 0.0)
                {
                    dataCT_->sign_ *= -1;
                }

                //The update matrices of size k-1 x k-1 with the pp row and col deleted and the last row and col now at index pp

                const size_t kk = dataCT_->vertices_.size();
                const size_t kkm1 = kk - 1;

                LinAlg::BlockDowngrade(nfdata_.N_, 2 * pp, 2);

                nfdata_.F_.swap_rows(2 * pp, 2 * kk - 2);
                nfdata_.F_.swap_rows(2 * pp + 1, 2 * kk - 1);
                nfdata_.F_.resize(2 * kkm1);

                std::iter_swap(dataCT_->vertices_.begin() + pp, dataCT_->vertices_.begin() + kk - 1);
                dataCT_->vertices_.pop_back();
                AssertSizes();
            }
        }
    }

    void CleanUpdate(bool print = false)
    {
        mpiUt::Print("Cleaning, sign, k =  " + std::to_string(dataCT_->sign_) + ",  " + std::to_string(dataCT_->vertices_.size()));
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

                nfdata_.N_(2 * i, 2 * j) = -GetGreenTau0Up(dataCT_->vertices_.at(i), dataCT_->vertices_.at(j)) * (nfdata_.F_(2 * j) - 1.0);               //Up Up Normal
                nfdata_.N_(2 * i, 2 * j + 1) = -GetFTau0UpDown(dataCT_->vertices_.at(i), dataCT_->vertices_.at(j));                                       //Up Down Anormal
                nfdata_.N_(2 * i + 1, 2 * j) = -GetFTau0DownUp(dataCT_->vertices_.at(i), dataCT_->vertices_.at(j));                                       //Down Up Anormal
                nfdata_.N_(2 * i + 1, 2 * j + 1) = -GetGreenTau0Down(dataCT_->vertices_.at(i), dataCT_->vertices_.at(j)) * (nfdata_.F_(2 * j + 1) - 1.0); //Down Down Normal

                if (i == j)
                {
                    nfdata_.N_(2 * i, 2 * i) += nfdata_.F_(2 * i);
                    nfdata_.N_(2 * i + 1, 2 * i + 1) += nfdata_.F_(2 * i + 1);
                }
            }
        }
        AssertSizes();

        nfdata_.N_.Inverse();
    }

    double GetGreenTau0Up(const Vertex &vertexI, const Vertex &vertexJ) const
    {
        return (dataCT_->green0CachedUp_(vertexI.site(), vertexJ.site(), vertexI.tau() - vertexJ.tau()));
    }

    // In fact return -g_Down(-tau) (nambu version of gdown)
    double GetGreenTau0Down(const Vertex &vertexI, const Vertex &vertexJ) const
    {

#ifdef AFM
        return (dataCT_->green0CachedDown_(vertexI.site(), vertexJ.site(), (vertexI.tau() - vertexJ.tau())));
#else
        return (dataCT_->green0CachedUp_(vertexI.site(), vertexJ.site(), (vertexI.tau() - vertexJ.tau())));

#endif
    }

    double GetFTau0DownUp(const Vertex &vertexI, const Vertex &vertexJ) const
    {
        return 0.0;
    }

    double GetFTau0UpDown(const Vertex &vertexI, const Vertex &vertexJ) const
    {
        return 0.0;
    }

    void Measure()
    {
        SiteVector_t FVM1 = -(nfdata_.F_ - 1.0);
        DDMGMM(FVM1, nfdata_.N_, *(dataCT_->MPtr_));
        // obs_.Measure();
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

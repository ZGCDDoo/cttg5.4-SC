#pragma once

#include "Integrator.hpp"
#include "Utilities.hpp"
#include "MPIUtilities.hpp"
#include "NambuMat.hpp"
#include "ABC_SelfConsistency.hpp"

namespace SelfCon
{

template <typename TH0>
struct GreenLattice
{

  public:
    static const size_t Nc;
    static const ClusterMatrixCD_t II;

    GreenLattice(cd_t zz, ClusterMatrixCD_t selfEnergy, TH0 h0) : zz_(zz), selfEnergy_(selfEnergy), h0_(h0){};

    ClusterMatrixCD_t operator()(const double &kx, const double &ky)
    {
        return ((zz_ * II - h0_(kx, ky) - selfEnergy_).i());
    }

  private:
    const cd_t zz_;
    ClusterMatrixCD_t selfEnergy_;
    TH0 h0_;
};
template <typename TH0>
const ClusterMatrixCD_t GreenLattice<TH0>::II = ClusterMatrixCD_t(TH0::Nc, TH0::Nc).eye();

template <typename TH0>
const size_t GreenLattice<TH0>::Nc = TH0::Nc;

template <typename TIOModel, typename TModel, typename TH0>
class SelfConsistency : public ABC_SelfConsistency
{

  public:
    static const size_t Nc;
    static const ClusterMatrixCD_t II;
    static const ClusterMatrixCD_t IINambu;

    const double factNSelfCon = 2.0;
    const size_t hybSavePrecision = 10;

    SelfConsistency(const Json &jj, const TModel &model, const ClusterCubeCD_t &nambuImpurity) : model_(model),
                                                                                                 ioModel_(TIOModel()),
                                                                                                 nambuImpurity_(nambuImpurity),
                                                                                                 hybridization_(model_.hybridizationMat()),
                                                                                                 selfEnergy_(),
                                                                                                 hybNext_(),
                                                                                                 weights_(jj["WEIGHTSR"].get<double>(), jj["WEIGHTSI"].get<double>()),
                                                                                                 breakSymmetry_(jj["BREAK_SYMMETRY"].get<bool>())
    {

        mpiUt::Print("Start of SC constructor");

        const size_t NGreen = nambuImpurity_.n_slices;
        size_t NSelfConTmp = std::max<double>(0.5 * (jj["ESelfCon"].get<double>() * model_.beta() / M_PI - 1.0),
                                              0.5 * (200.0 * model_.beta() / M_PI - 1.0));
        if (NGreen >= NSelfConTmp)
        {
            NSelfConTmp = factNSelfCon * static_cast<double>(NGreen);
        }
        const size_t NSelfCon = NGreen; //NSelfConTmp;
        // assert(NSelfCon > NGreen);
        //Patcher la hyb si necessaire
        hybridization_.PatchHF(NSelfCon, model_.beta());
        const size_t NHyb = hybridization_.n_slices();
        assert(NHyb >= NSelfCon);

        const size_t NNambu = 2 * Nc;
        selfEnergy_.resize(NNambu, NNambu, NSelfCon);
        selfEnergy_.zeros();

        //0.) Extraire la self jusqu'a NGreen
        const ClusterMatrixCD_t II2x2Nambu = {{cd_t(1.0), cd_t(0.0)}, {cd_t(0.0), cd_t(-1.0)}};

        const ClusterMatrixCD_t II = ClusterMatrixCD_t(NNambu, NNambu).eye();
        const ClusterMatrixCD_t IINambu = arma::kron(II2x2Nambu, ClusterMatrixCD_t(Nc, Nc).eye());

        const ClusterMatrixCD_t tLocNambu = arma::kron(II2x2Nambu, model_.tLoc());
        const ClusterMatrixCD_t muNambu = model_.mu() * IINambu;

        const ClusterMatrix_t II2x2Off = {{0.0, 1.0}, {1.0, 0.0}};
        const ClusterMatrixCD_t SM = 1e-2 * ClusterMatrixCD_t(arma::kron(II2x2Off, ioModel_.signFAnormal()), ClusterMatrix_t(NNambu, NNambu).zeros());

        for (size_t nn = 0; nn < NGreen; nn++)
        {
            const cd_t iwn(0.0, (2.0 * nn + 1.0) * M_PI / model_.beta());
            selfEnergy_.slice(nn) = -nambuImpurity_.slice(nn).i() + iwn * II + muNambu - tLocNambu - hybridization_.slice(nn);
            if (breakSymmetry_)
            {
                const double wn = iwn.imag();
                selfEnergy_.slice(nn) += SM / (1.0 + wn * wn);
            }
        }

        //1.) Patcher la self par HF de NGreen à NSelfCon
        const ClusterMatrix_t II2x2 = ClusterMatrix_t(2, 2).eye();

        ClusterMatrix_t nUpMatrix;
        assert(nUpMatrix.load("nUpMatrix.dat"));
        nUpMatrix = arma::kron(II2x2, nUpMatrix);

        ClusterMatrix_t nDownMatrix;
        assert(nDownMatrix.load("nDownMatrix.dat"));
        nDownMatrix = arma::kron(II2x2, nDownMatrix);

        ClusterMatrixCD_t nMatrix(nUpMatrix + nDownMatrix, ClusterMatrix_t(NNambu, NNambu).zeros());

        for (size_t nn = NGreen; nn < NSelfCon; nn++)
        {
            const cd_t iwn(0.0, (2.0 * nn + 1.0) * M_PI / model_.beta());
#ifndef AFM
            selfEnergy_.slice(nn) = 0.5 * model_.U() * IINambu * nMatrix + 1.0 / iwn * model_.U() * model_.U() * nMatrix / 2.0 * (II - nMatrix / 2.0);
#else
            if (spin_ == FermionSpin_t::Up)
            {
                selfEnergy_.slice(nn) = model_.U() * nDownMatrix + 1.0 / iwn * model_.U() * model_.U() * nDownMatrix * (IINambu - nDownMatrix);
            }
            else if (spin_ == FermionSpin_t::Down)
            {
                selfEnergy_.slice(nn) = model_.U() * nUpMatrix + 1.0 / iwn * model_.U() * model_.U() * nUpMatrix * (IINambu - nUpMatrix);
            }
            else
            {
                throw std::runtime_error("Ayaya, must be a spin man");
            }
#endif
        }

        if (mpiUt::Rank() == mpiUt::master)
        {
            ioModel_.SaveCube("self", selfEnergy_, model_.beta(), hybSavePrecision);
            std::cout << "In Selfonsistency constructor, after save selfenery " << std::endl;
        }

        mpiUt::Print("After SC constructor");
    }

    void DoSCGrid() override
    {
        // #ifdef HAVEMPI
        // DoSCGridParallel();
        // #else
        DoSCGridSerial();
        // #endif
    }

    // #ifdef HAVEMPI
    //     void DoSCGridParallel()
    //     {

    //         mpi::communicator world;

    //         mpiUt::Print("In Selfonsistency DOSC Parallel");
    //         const size_t NSelfCon = selfEnergy_.n_slices;

    //         if (static_cast<size_t>(mpiUt::NWorkers()) > NSelfCon)
    //         {
    //             DoSCGridSerial();
    //             return;
    //         }

    //         const size_t NSelfConRank = mpiUt::Rank() == mpiUt::master ? (NSelfCon / mpiUt::NWorkers() + NSelfCon % mpiUt::NWorkers()) : NSelfCon / mpiUt::NWorkers();

    //         ClusterCubeCD_t gImpUpNextRank(Nc, Nc, NSelfConRank);
    //         gImpUpNextRank.zeros();
    //         ClusterCubeCD_t hybNextRank(Nc, Nc, NSelfConRank);
    //         hybNextRank.zeros();

    //         ClusterCubeCD_t tKTildeGrid;
    //         assert(tKTildeGrid.load("tktilde.arma"));
    //         const size_t ktildepts = tKTildeGrid.n_slices;

    //         const size_t nnStart = mpiUt::Rank() == mpiUt::master ? 0 : NSelfCon % mpiUt::NWorkers() + (NSelfCon / mpiUt::NWorkers()) * mpiUt::Rank();
    //         const size_t nnEnd = nnStart + NSelfConRank;
    //         for (size_t nn = nnStart; nn < nnEnd; nn++)
    //         {
    //             const cd_t zz = cd_t(model_.mu(), (2.0 * nn + 1.0) * M_PI / model_.beta());
    //             for (size_t ktildeindex = 0; ktildeindex < ktildepts; ktildeindex++)
    //             {
    //                 gImpUpNextRank.slice(nn - nnStart) += 1.0 / (static_cast<double>(ktildepts)) * ((zz * ClusterMatrixCD_t(Nc, Nc).eye() - tKTildeGrid.slice(ktildeindex) - selfEnergy_.slice(nn)).i());
    //             }
    //             hybNextRank.slice(nn - nnStart) = -gImpUpNextRank.slice(nn - nnStart).i() - selfEnergy_.slice(nn) + zz * ClusterMatrixCD_t(Nc, Nc).eye() - model_.tLoc();
    //         }

    //         std::vector<std::vector<cd_t>> tmpMemGImpVec;
    //         std::vector<std::vector<cd_t>> tmpMemHybNextVec;
    //         std::vector<cd_t> tmpMemGImp = mpiUt::CubeCDToVecCD(gImpUpNextRank);
    //         std::vector<cd_t> tmpMemHybNext = mpiUt::CubeCDToVecCD(hybNextRank);

    //         if (mpiUt::Rank() == mpiUt::master)
    //         {
    //             mpi::gather(world, tmpMemGImp, tmpMemGImpVec, mpiUt::master);
    //             mpi::gather(world, tmpMemHybNext, tmpMemHybNextVec, mpiUt::master);
    //         }
    //         else
    //         {
    //             mpi::gather(world, tmpMemGImp, mpiUt::master);
    //             mpi::gather(world, tmpMemHybNext, mpiUt::master);
    //         }

    //         if (mpiUt::Rank() == mpiUt::master)
    //         {
    //             ClusterCubeCD_t gImpUpNext(Nc, Nc, NSelfCon);
    //             gImpUpNext.zeros();
    //             hybNext_.resize(Nc, Nc, NSelfCon);
    //             hybNext_.zeros();

    //             for (size_t ii = 0; ii < static_cast<size_t>(mpiUt::NWorkers()); ii++)
    //             {
    //                 ClusterCubeCD_t tmpGImpNextRank = mpiUt::VecCDToCubeCD(tmpMemGImpVec.at(ii), Nc, Nc, tmpMemGImpVec.at(ii).size() / (Nc * Nc));
    //                 ClusterCubeCD_t tmpHybNextRank = mpiUt::VecCDToCubeCD(tmpMemHybNextVec.at(ii), Nc, Nc, tmpMemHybNextVec.at(ii).size() / (Nc * Nc));

    //                 const size_t jjStart = ii == 0 ? 0 : NSelfCon % mpiUt::NWorkers() + (NSelfCon / mpiUt::NWorkers()) * ii;
    //                 const size_t jjEnd = jjStart + tmpGImpNextRank.n_slices;
    //                 for (size_t jj = jjStart; jj < jjEnd; jj++)
    //                 {
    //                     gImpUpNext.slice(jj) = tmpGImpNextRank.slice(jj - jjStart);
    //                     hybNext_.slice(jj) = tmpHybNextRank.slice(jj - jjStart);
    //                 }
    //             }

    //             hybNext_ *= (1.0 - weights_);
    //             hybNext_ += weights_ * hybridization_.data();
    //             ioModel_.SaveCube("green", gImpUpNext, model_.beta(), hybSavePrecision);
    //             ioModel_.SaveCube("hybNext" + GetSpinName(spin_), hybNext_, model_.beta(), hybSavePrecision);

    //             mpiUt::Print("After Selfonsistency DOSC Parallel");
    //         }
    //     }

    // #endif

    void DoSCGridSerial()
    {

        if (mpiUt::Rank() == mpiUt::master)
        {
            std::cout << "In Selfonsistency DOSC serial" << std::endl;
            const size_t NSelfCon = selfEnergy_.n_slices;
            const size_t NNambu = 2 * Nc;
            ClusterCubeCD_t gImpUpNext(NNambu, NNambu, NSelfCon);
            gImpUpNext.zeros();
            hybNext_.resize(NNambu, NNambu, NSelfCon);
            hybNext_.zeros();
            ClusterCubeCD_t tKTildeGrid;
            assert(tKTildeGrid.load("tktilde.arma"));
            const size_t ktildepts = tKTildeGrid.n_slices;

            const ClusterMatrixCD_t II2 = ClusterMatrixCD_t(2, 2).eye();
            const ClusterMatrixCD_t II2x2Nambu = {{cd_t(1.0), cd_t(0.0)}, {cd_t(0.0), cd_t(-1.0)}};

            const ClusterMatrixCD_t II = ClusterMatrixCD_t(NNambu, NNambu).eye();
            const ClusterMatrixCD_t IINambu = arma::kron(II2x2Nambu, ClusterMatrixCD_t(Nc, Nc).eye());

            const ClusterMatrixCD_t tLocNambu = arma::kron(II2x2Nambu, model_.tLoc());
            const ClusterMatrixCD_t muNambu = model_.mu() * IINambu;

            std::cout << "Here 1" << std::endl;

            for (size_t nn = 0; nn < NSelfCon; nn++)
            {
                const cd_t iwn(0.0, (2.0 * static_cast<double>(nn) + 1.0) * M_PI / model_.beta());
                for (size_t ktildeindex = 0; ktildeindex < ktildepts; ktildeindex++)
                {
                    const ClusterMatrixCD_t tkTildeGridNambu = arma::kron(II2x2Nambu, tKTildeGrid.slice(ktildeindex));
                    gImpUpNext.slice(nn) += (iwn * II + muNambu - tkTildeGridNambu - selfEnergy_.slice(nn)).i();
                }
                gImpUpNext.slice(nn) /= static_cast<double>(ktildepts);
                hybNext_.slice(nn) = -gImpUpNext.slice(nn).i() - selfEnergy_.slice(nn) + iwn * II + muNambu - tLocNambu;
            }
            std::cout << "Here 2 " << std::endl;

            hybNext_ *= (1.0 - weights_);
            hybNext_ += weights_ * hybridization_.data();
            ioModel_.SaveCube("green", gImpUpNext, model_.beta(), hybSavePrecision);
            ioModel_.SaveCube("hybNext", hybNext_, model_.beta(), hybSavePrecision);

            std::cout << "After Selfonsistency DOSC serial" << std::endl;
        }
    }

    ClusterCubeCD_t
    hybNext() const
    {
        return hybNext_;
    };

  private:
    TModel model_;
    TIOModel ioModel_;

    const ClusterCubeCD_t nambuImpurity_;
    NambuMat::HybridizationMat hybridization_;
    ClusterCubeCD_t selfEnergy_;
    ClusterCubeCD_t hybNext_;
    const cd_t weights_;
    const bool breakSymmetry_;
};
template <typename TIOModel, typename TModel, typename TH0>
const ClusterMatrixCD_t SelfConsistency<TIOModel, TModel, TH0>::II = ClusterMatrixCD_t(TH0::Nc, TH0::Nc).eye();

template <typename TIOModel, typename TModel, typename TH0>
const ClusterMatrixCD_t SelfConsistency<TIOModel, TModel, TH0>::IINambu = ClusterMatrixCD_t(2 * TH0::Nc, 2 * TH0::Nc).eye();

template <typename TIOModel, typename TModel, typename TH0>
const size_t SelfConsistency<TIOModel, TModel, TH0>::Nc = TH0::Nc;

} // namespace SelfCon
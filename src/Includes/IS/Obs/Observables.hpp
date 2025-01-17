#pragma once

#include "../../Utilities/Utilities.hpp"
#include "../../Utilities/LinAlg.hpp"
#include "../../Utilities/MPIUtilities.hpp"

#include "GreenBinning.hpp"
#include "FillingAndDocc.hpp"
// #include "KineticEnergy.hpp"
#include "../ISData.hpp"
#include "../ISResult.hpp"

namespace Markov
{
namespace Obs
{
using Matrix_t = LinAlg::Matrix<double>;

template <typename TIOModel, typename TModel>
class Observables
{

      public:
        Observables(){};
        Observables(const std::shared_ptr<ISDataCT<TIOModel, TModel>> &dataCT,
                    const Json &jj) : modelPtr_(new TModel(jj)),
                                      ioModel_(TIOModel()),
                                      dataCT_(dataCT),
                                      rng_(jj["SEED"].get<size_t>() + mpiUt::Rank() * mpiUt::Rank()),
                                      urngPtr_(new Utilities::UniformRngFibonacci3217_t(rng_, Utilities::UniformDistribution_t(0.0, 1.0))),
                                      greenBinning_(modelPtr_, dataCT_, jj, FermionSpin_t::Up),
                                      fillingAndDocc_(dataCT_, urngPtr_, jj["N_T_INV"].get<size_t>()),
                                      signMeas_(0.0),
                                      expOrder_(0.0),
                                      NMeas_(0)
        {
                mpiUt::Print("After Obs constructor ");
        }

        //Getters
        double signMeas() const { return signMeas_; };
        double expOrder() const { return expOrder; };

        void Measure()
        {

                // mpiUt::Print("start of Measure");

                NMeas_++;
                signMeas_ += static_cast<double>(dataCT_->sign_);
                expOrder_ += static_cast<double>(dataCT_->vertices_.size()) * static_cast<double>(dataCT_->sign_);
                fillingAndDocc_.MeasureFillingAndDocc();
                greenBinning_.MeasureGreenBinning(*dataCT_->MPtr_);

                // mpiUt::Print("End of Measure");
        }

        void Save()
        {
                mpiUt::Print("Start of Observables.Save()");
                signMeas_ /= NMeas_;

                fillingAndDocc_.Finalize(signMeas_, NMeas_);
                std::map<std::string, double> obsScal;

                obsScal = fillingAndDocc_.GetObs();

                obsScal["sign"] = signMeas_;
                obsScal["NMeas"] = NMeas_;

                //dont forget that the following obs have not been finalized (multiplied by following factor)
                const double fact = 1.0 / (NMeas_ * signMeas_);
                obsScal["k"] = fact * expOrder_;

                const ClusterCubeCD_t greenNambuCube = greenBinning_.FinalizeGreenBinning(signMeas_, NMeas_);

                //Gather and stats of all the results for all cores
                Result::ISResult isResult(obsScal, greenNambuCube, fillingAndDocc_.fillingUp(), fillingAndDocc_.fillingDown());
                std::vector<Result::ISResult> isResultVec;
#ifdef HAVEMPI

                mpi::communicator world;
                if (mpiUt::Rank() == mpiUt::master)
                {
                        mpi::gather(world, isResult, isResultVec, mpiUt::master);
                }
                else
                {
                        mpi::gather(world, isResult, mpiUt::master);
                }
                if (mpiUt::Rank() == mpiUt::master)
                {
                        mpiUt::IOResult<TIOModel>::SaveISResults(isResultVec, dataCT_->beta_);
                }

#else
                isResultVec.push_back(isResult);
                mpiUt::IOResult<TIOModel>::SaveISResults(isResultVec, dataCT_->beta_);
#endif

                // Start: This should be in PostProcess.cpp ?
                //Start of observables that are easier and ok to do once all has been saved (for exemples, depends only on final green function)
                //Get KinecticEnergy
                // #ifndef DCA
                //                 if (mpiUt::Rank() == mpiUt::master)
                //                 {
                //                         std::ifstream fin("Obs.json");
                //                         Json results;
                //                         fin >> results;
                //                         fin.close();

                //                         std::cout << "Start Calculating Kinetic Energy " << std::endl;
                //                         KineticEnergy<TModel, TIOModel> kEnergy(modelPtr_, ioModel_.ReadGreenDat("greenUp.dat"));
                //                         results["KEnergy"] = {kEnergy.GetKineticEnergy(), 0.0};
                //                         std::cout << "End Calculating Kinetic Energy " << std::endl;

                //                         std::ofstream fout("Obs.json");
                //                         fout << std::setw(4) << results << std::endl;
                //                         fout.close();
                //                 }

                //                 //End: This should be in PostProcess.cpp ?
                // #endif
                //                 //ioModel_.SaveCube("greenUp.dat", modelPtr_->greenCluster0MatUp().data(), modelPtr_->beta());
                mpiUt::Print("End of Observables.Save()");
        }

      private:
        std::shared_ptr<TModel> modelPtr_;
        TIOModel ioModel_;
        std::shared_ptr<ISDataCT<TIOModel, TModel>> dataCT_;
        Utilities::EngineTypeFibonacci3217_t rng_;
        std::shared_ptr<Utilities::UniformRngFibonacci3217_t> urngPtr_;

        GreenBinning<TIOModel, TModel> greenBinning_;
        FillingAndDocc<TIOModel, TModel> fillingAndDocc_;

        //=======Measured quantities
        double signMeas_;
        double expOrder_;

        size_t NMeas_;
};

} // namespace Obs
} // namespace Markov

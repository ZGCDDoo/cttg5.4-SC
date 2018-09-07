#pragma once

#include "../Utilities/Utilities.hpp"
#include "../Utilities/MPIUtilities.hpp"
#include "ABC_MonteCarlo.hpp"
#include <chrono>
#include <ctime>

namespace MC
{

struct Timer
{
    void Start(double duration)
    {
        duration_ = duration;
        start_ = std::chrono::steady_clock::now();
    };

    void PrintTime()
    {
        if (mpiUt::Rank() == 0)
        {
            auto timeChrono = std::chrono::system_clock::now();
            std::time_t timeNow = std::chrono::system_clock::to_time_t(timeChrono);
            std::cout << "\t " << std::ctime(&timeNow) << std::endl;
        }

        return;
    }

    bool End()
    {
        return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_).count() > duration_;
    };

  private:
    double duration_;
    std::chrono::steady_clock::time_point start_;
};

template <typename TMarkovChain_t>
class MonteCarlo : public ABC_MonteCarlo
{
  public:
    MonteCarlo(const std::shared_ptr<TMarkovChain_t> &markovchainPtr, const Json &jj) : markovchainPtr_(markovchainPtr),
                                                                                        thermalizationTime_(jj["THERMALIZATION_TIME"].get<double>()),
                                                                                        measurementTime_(jj["MEASUREMENT_TIME"].get<double>()),
#if defined SUBMATRIX
                                                                                        updatesMeas_((jj["UPDATESMEAS"].get<size_t>() / jj["KMAX_UPD"].get<size_t>()) * jj["KMAX_UPD"].get<size_t>()),
#else
                                                                                        updatesMeas_(jj["UPDATESMEAS"].get<size_t>()),
#endif
                                                                                        cleanUpdate_(jj["CLEANUPDATE"].get<size_t>()),
                                                                                        NMeas_(0),
                                                                                        NCleanUpdates_(0),
                                                                                        thermFromConfig_(jj["THERM_FROM_CONFIG"].get<bool>())
    {
    }

    ~MonteCarlo(){};

    void RunMonteCarlo()
    {
        auto time = std::chrono::system_clock::now();
        Timer timer;

        if (thermFromConfig_)
        {
            markovchainPtr_->ThermalizeFromConfig();
        }
        else
        {

            time = std::chrono::system_clock::now();
            mpiUt::Print(std::string("Start Thermalization at: "));
            timer.PrintTime();

            timer.Start(60.0 * thermalizationTime_);
            while (true)
            {
                markovchainPtr_->DoStep();
                if (markovchainPtr_->updatesProposed() % updatesMeas_ == 0)
                {
                    if (timer.End())
                    {
                        break;
                    }
                    NMeas_++;
                    if (NMeas_ % cleanUpdate_ == 0 && NMeas_ != 0)
                    {
                        markovchainPtr_->CleanUpdate();
                    }
                }
            }

            markovchainPtr_->SaveTherm();
            mpiUt::Print(std::string("End Thermalization at: "));
            timer.PrintTime();
        }

        NMeas_ = 0;
        timer.Start(60.0 * measurementTime_);
        mpiUt::Print(std::string("Start Measurements at: "));
        timer.PrintTime();

        while (true)
        {
            markovchainPtr_->DoStep(); //One simple sweep

            if (markovchainPtr_->updatesProposed() % updatesMeas_ == 0)
            {
                if (timer.End())
                {
                    break;
                }
                markovchainPtr_->Measure();
                NMeas_++;
                //mpiUt::Print(std::string("Measuring, ") + std::to_string(updatesProposed_) + std::string(" Updates Proposed"));
                if (NMeas_ % cleanUpdate_ == 0 && NMeas_ != 0)
                {
                    markovchainPtr_->CleanUpdate();
                    NCleanUpdates_++;
                    //mpiUt::Print(std::string("Cleaning, ") + std::to_string(updatesProposed_) + std::string(" Updates Proposed"));
                }
            }
        }

        mpiUt::Print(std::string("End Measurements at: "));
        timer.PrintTime();
        markovchainPtr_->SaveMeas();
        return;
    }

    //Getters
    size_t NMeas() const { return NMeas_; };
    size_t NCleanUpdates() const { return NCleanUpdates_; };
    size_t updatesProposed() const { return markovchainPtr_->updatesProposed(); };

  private:
    //attributes
    const std::shared_ptr<TMarkovChain_t> markovchainPtr_;
    const double thermalizationTime_;
    const double measurementTime_;
    const size_t updatesMeas_;
    const size_t cleanUpdate_;

    size_t NMeas_;
    size_t NCleanUpdates_;
    bool thermFromConfig_;
};
} // namespace MC

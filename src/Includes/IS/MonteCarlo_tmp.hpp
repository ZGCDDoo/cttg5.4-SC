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
class MonteCarlo : public ABC_MonteCarlo<TMarkovChain_t>
{
  public:
    MonteCarlo(const std::unique_ptr<TMarkovChain_t> &markovchainPtr, const Json &jj)
        : ABC_MonteCarlo<TMarkovChain_t>(markovchainPtr, jj) {}

    ~MonteCarlo(){};

    void RunMonteCarlo() override
    {
        auto time = std::chrono::system_clock::now();
        Timer timer;

        if (this->thermFromConfig_)
        {
            this->markovchainPtr_->ThermalizeFromConfig();
        }
        else
        {

            time = std::chrono::system_clock::now();
            mpiUt::Print(std::string("Start Thermalization at: "));
            timer.PrintTime();

            timer.Start(60.0 * this->thermalizationTime_);
            while (true)
            {
                this->markovchainPtr_->DoStep();
                if (this->markovchainPtr_->updatesProposed() % this->updatesMeas_ == 0)
                {
                    if (timer.End())
                    {
                        break;
                    }
                    this->NMeas_++;
                    if (this->NMeas_ % this->cleanUpdate_ == 0 && this->NMeas_ != 0)
                    {
                        this->markovchainPtr_->CleanUpdate();
                    }
                }
            }

            this->markovchainPtr_->SaveTherm();
            mpiUt::Print(std::string("End Thermalization at: "));
            timer.PrintTime();
        }

        this->NMeas_ = 0;
        timer.Start(60.0 * this->measurementTime_);
        mpiUt::Print(std::string("Start Measurements at: "));
        timer.PrintTime();

        while (true)
        {
            this->markovchainPtr_->DoStep(); //One simple sweep

            if (this->markovchainPtr_->updatesProposed() % this->updatesMeas_ == 0)
            {
                if (timer.End())
                {
                    break;
                }
                this->markovchainPtr_->Measure();
                this->NMeas_++;
                //mpiUt::Print(std::string("Measuring, ") + std::to_string(updatesProposed_) + std::string(" Updates Proposed"));
                if (this->NMeas_ % this->cleanUpdate_ == 0 && this->NMeas_ != 0)
                {
                    this->markovchainPtr_->CleanUpdate();
                    this->NCleanUpdates_++;
                    //mpiUt::Print(std::string("Cleaning, ") + std::to_string(updatesProposed_) + std::string(" Updates Proposed"));
                }
            }
        }

        mpiUt::Print(std::string("End Measurements at: "));
        timer.PrintTime();
        this->markovchainPtr_->SaveMeas();
    }

    //Getters

  private:
};
} // namespace MC

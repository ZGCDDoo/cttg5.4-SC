#pragma once

#include "../../Utilities/Utilities.hpp"
#include "../../Utilities/LinAlg.hpp"
#include "../ISData.hpp"

namespace Markov
{
namespace Obs
{

template <typename TIOModel, typename TModel>
class GreenTauMesure
{

  public:
    const size_t N_T_INV = 5;
    GreenTauMesure(const std::shared_ptr<ISDataCT<TIOModel, TModel>> &dataCT,
                   std::shared_ptr<Utilities::UniformRngFibonacci3217_t> urngPtr) : dataCT_(dataCT),
                                                                                    ioModel_(TIOModel()),
                                                                                    urngPtr_(urngPtr)
    {
        greenUpCurrent_.resize(ioModel_.indepSites().size());
        greenDownCurrent_.resize(ioModel_.indepSites().size());
        greenUpCurrent_.zeros();
        greenDownCurrent_.zeros();
    }

    SiteVector_t greenUpCurrent() const { return greenUpCurrent_; }
    SiteVector_t greenDownCurrent() const { return greenDownCurrent_; }

    //measure green with N_T_INV estimates
    void MeasureGreen(const double &tau)
    {

        // mpiUt::Print("start of GreenTauMeasure.MeasureGreen");

        const size_t KK = dataCT_->vertices_.size();

        greenUpCurrent_.zeros();
        greenDownCurrent_.zeros();

        // for futur use with AFM ?
        // std::shared_ptr<GreenTau> greenCached(spin == FermionSpin_t::Up ? &(dataCT_->green0CachedUp_) : &(dataCT_->green0CachedUp_));
        SiteVector_t vec1(KK);
        SiteVector_t vec2(KK);

        const double sign = static_cast<double>(dataCT_->sign_);

        for (size_t ii = 0; ii < ioModel_.indepSites().size(); ii++)
        {
            Site_t s1 = ioModel_.indepSites().at(ii).first;
            Site_t s2 = ioModel_.indepSites()[ii].second;

            double dotUp = 0.0;
            double dotDown = 0.0;

            if (KK)
            {
                for (size_t nsamples = 0; nsamples < N_T_INV; nsamples++)
                {
                    //tau  = tauRng - tauRng2
                    const double tauRng = 0.0; //(*urngPtr_)() * dataCT_->beta_;
                    // double tauRng2 = tauRng - tau;

                    for (size_t kk = 0; kk < KK; kk++)
                    {
                        Site_t rr = dataCT_->vertices_[kk].site();
                        Tau_t tt = dataCT_->vertices_.at(kk).tau();

                        // assert(std::abs(tauRng - tt) <= dataCT_->beta_);
                        // assert(std::abs(tauRng2 - tt) <= dataCT_->beta_);

                        // std::cout << "tau + tauRng - tt = " << tau + tauRng - tt << std::endl;
                        // std::cout << "tt - tauRng = " << tt - tauRng << std::endl;

                        vec1(kk) = dataCT_->green0CachedUp_(s1, rr, tau + tauRng - tt);
                        vec2(kk) = dataCT_->green0CachedUp_(rr, s2, tt - tauRng);
                    }

                    dotUp += LinAlg::Dot(vec1, *(dataCT_->MupPtr_), vec2);
                    dotDown += LinAlg::Dot(vec1, *(dataCT_->MdownPtr_), vec2);
                }
            }

            const double green0 = dataCT_->green0CachedUp_(s1, s2, tau);
            greenUpCurrent_(ii) = sign * (green0 - dotUp / static_cast<double>(N_T_INV));
            greenDownCurrent_(ii) = sign * (green0 - dotDown / static_cast<double>(N_T_INV));
        }

        // mpiUt::Print("End of GreenTauMeasure.MeasureGreen");
        return;
    }

  private:
    const std::shared_ptr<ISDataCT<TIOModel, TModel>> &dataCT_;
    TIOModel ioModel_;
    std::shared_ptr<Utilities::UniformRngFibonacci3217_t> urngPtr_;
    SiteVector_t greenUpCurrent_;   //for current config
    SiteVector_t greenDownCurrent_; //for current config
};

} // namespace Obs
} // namespace Markov
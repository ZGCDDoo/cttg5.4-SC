#pragma once

#include "../../Utilities/Utilities.hpp"
#include "../../Utilities/LinAlg.hpp"
#include "../ISData.hpp"

namespace Markov
{
namespace Obs
{

template <typename TIOModel, typename TModel>
class FillingAndDocc
{

  public:
    FillingAndDocc(const std::shared_ptr<ISDataCT<TIOModel, TModel>> &dataCT,
                   std::shared_ptr<Utilities::UniformRngFibonacci3217_t> urngPtr, const size_t &N_T_INV) : dataCT_(dataCT),
                                                                                                           ioModel_(TIOModel()),
                                                                                                           urngPtr_(urngPtr),
                                                                                                           N_T_INV_(N_T_INV)
    {
        const size_t LL = ioModel_.indepSites().size();

        fillingUpCurrent_.resize(LL, 0.0);
        fillingDownCurrent_.resize(LL, 0.0);
        doccCurrent_.resize(LL, 0.0);
        SzCurrent_.resize(LL, 0.0);

        fillingUp_.resize(LL, 0.0);
        fillingDown_.resize(LL, 0.0);
        docc_.resize(LL, 0.0);
        Sz_.resize(LL, 0.0);
    }

    std::vector<double> fillingUp() const { return fillingUp_; }
    std::vector<double> fillingDown() const { return fillingDown_; }
    std::valarray<double> fillingDownCurrent() const { return fillingDownCurrent_; }
    std::valarray<double> fillingUpCurrent() const { return fillingUpCurrent_; }
    std::map<std::string, double> GetObs() const { return obsmap_; }

    double fillingUpTotalCurrent()
    {
        double fillingUpTotalCurrent = 0.0;

        for (size_t kk = 0; kk < ioModel_.fillingSites().size(); kk++)
        {

            const double factFilling = static_cast<double>(ioModel_.nOfAssociatedSites().at(ioModel_.fillingSitesIndex().at(kk)));
            fillingUpTotalCurrent += factFilling * fillingUpCurrent_[kk];
        }
        return fillingUpTotalCurrent;
    };

    double fillingDownTotalCurrent()
    {
        double fillingDownTotalCurrent = 0.0;

        for (size_t kk = 0; kk < ioModel_.fillingSites().size(); kk++)
        {

            const double factFilling = static_cast<double>(ioModel_.nOfAssociatedSites().at(ioModel_.fillingSitesIndex().at(kk)));
            fillingDownTotalCurrent += factFilling * fillingDownCurrent_[kk];
        }
        return fillingDownTotalCurrent;
    };

    double doccTotalCurrent()
    {
        double doccTotalCurrent = 0.0;

        for (size_t kk = 0; kk < ioModel_.fillingSites().size(); kk++)
        {

            const double factFilling = static_cast<double>(ioModel_.nOfAssociatedSites().at(ioModel_.fillingSitesIndex().at(kk)));
            doccTotalCurrent += factFilling * doccCurrent_[kk];
        }
        return doccTotalCurrent;
    };

    void ResetCurrent()
    {
        doccCurrent_ = 0.0;
        SzCurrent_ = 0.0;
        fillingUpCurrent_ = 0.0;
        fillingDownCurrent_ = 0.0;
    }

    void MeasureFillingAndDocc()
    {
        // mpiUt::Print("start of MeasureFillingAndDocc");
        ResetCurrent();
        // mpiUt::Print("After resetcurrent");

        const size_t KK = dataCT_->vertices_.size();

        const double eps = 1e-12;

        SiteVector_t vec1Up(KK);
        SiteVector_t vec2Up(KK);
#ifdef AFM
        SiteVector_t vec1Down(KK);
        SiteVector_t vec2Down(KK);
#endif
        const double sign = static_cast<double>(dataCT_->sign_);
        const size_t fillingSize = ioModel_.fillingSites().size();

        for (Site_t ii = 0; ii < fillingSize; ii++)
        {
            const Site_t s1 = ioModel_.fillingSites()[ii];

            for (size_t nsamples = 0; nsamples < N_T_INV_; nsamples++)
            {
                const double tauRng = (*urngPtr_)() * dataCT_->beta_;
                const Site_t siteRng = ioModel_.FindSitesRng(s1, s1, (*urngPtr_)()).first;

                for (size_t kk = 0; kk < KK; kk++)
                {
                    const Site_t ss = dataCT_->vertices_.at(kk).site();
                    const Tau_t tt = dataCT_->vertices_[kk].tau();
                    vec1Up(kk) = dataCT_->green0CachedUp_(siteRng, ss, tauRng - tt);
                    vec2Up(kk) = dataCT_->green0CachedUp_(ss, siteRng, tt - tauRng);
#ifdef AFM
                    vec1Down(kk) = dataCT_->green0CachedDown_(siteRng, ss, tauRng - tt);
                    vec2Down(kk) = dataCT_->green0CachedDown_(ss, siteRng, tt - tauRng);
#endif
                }

                // mpiUt::Print("In loop before dots");
                double dotup = 0.0;
                double dotdown = 0.0;

                if (KK)
                {
                    dotup = LinAlg::Dot(vec1Up, *(dataCT_->MupPtr_), vec2Up);
#ifndef AFM
                    dotdown = LinAlg::Dot(vec1Up, *(dataCT_->MdownPtr_), vec2Up);
#endif
#ifdef AFM
                    dotdown = LinAlg::Dot(vec1Down, *(dataCT_->MdownPtr_), vec2Down);
#endif
                }

                const double green00Up = dataCT_->green0CachedUp_(s1, s1, -eps);
                double green00Down = green00Up;
#ifdef AFM
                green00Down = dataCT_->green0CachedDown_(s1, s1, -eps);
#endif
                const double nUptmp = green00Up - dotup;
                const double nDowntmp = green00Down - dotdown;
                fillingUpCurrent_[ii] += sign * nUptmp;
                fillingDownCurrent_[ii] += sign * nDowntmp;
                doccCurrent_[ii] += sign * (nUptmp * nDowntmp);
                // mpiUt::Print("here");

                const double ndiff = nUptmp - nDowntmp;
                SzCurrent_[ii] += sign * ndiff;
                // mpiUt::Print("here2");
            }
            // mpiUt::Print("here3");

            fillingUpCurrent_[ii] /= static_cast<double>(N_T_INV_);
            fillingDownCurrent_[ii] /= static_cast<double>(N_T_INV_);
            doccCurrent_[ii] /= static_cast<double>(N_T_INV_);
            SzCurrent_[ii] /= static_cast<double>(N_T_INV_);

            fillingUp_.at(ii) += fillingUpCurrent_[ii];
            fillingDown_[ii] += fillingDownCurrent_[ii];
            docc_[ii] += doccCurrent_[ii];
            Sz_[ii] += SzCurrent_[ii];
        }

        // mpiUt::Print("End of MeasureFillingAndDocc");
        return;
    }

    void Finalize(const double &signMeas, const double &NMeas)
    {

        const double fact = NMeas * signMeas;
        double fillingTotal = 0.0;
        double doccTotal = 0.0;
        double SzTotal = 0.0;
        std::string nName;

        for (size_t kk = 0; kk < ioModel_.fillingSites().size(); kk++)
        {
            fillingUp_.at(kk) /= fact;
            fillingDown_.at(kk) /= fact;

            nName = "n" + std::to_string(ioModel_.fillingSites().at(kk));
            obsmap_[nName + "Up"] = fillingUp_.at(kk);
            obsmap_[nName + "Down"] = fillingDown_.at(kk);

            docc_.at(kk) /= fact;
            nName = "docc" + std::to_string(ioModel_.fillingSites().at(kk));
            obsmap_[nName] = docc_.at(kk);

            Sz_.at(kk) /= fact;
            nName = "Sz" + std::to_string(ioModel_.fillingSites().at(kk));
            obsmap_[nName] = Sz_.at(kk);

            const double factFilling = static_cast<double>(ioModel_.nOfAssociatedSites().at(ioModel_.fillingSitesIndex().at(kk))) / static_cast<double>(ioModel_.Nc);
            fillingTotal += factFilling * (fillingUp_.at(kk) + fillingDown_.at(kk));
            doccTotal += factFilling * docc_.at(kk);
            SzTotal += factFilling * Sz_.at(kk);
        }

        obsmap_["n"] = fillingTotal;
        obsmap_["docc"] = doccTotal;
        obsmap_["Sz"] = SzTotal;
    }

  private:
    const std::shared_ptr<ISDataCT<TIOModel, TModel>> &dataCT_;
    TIOModel ioModel_;
    std::shared_ptr<Utilities::UniformRngFibonacci3217_t> urngPtr_;

    std::map<std::string, double> obsmap_;

    //current values for the current config
    std::valarray<double> fillingUpCurrent_;
    std::valarray<double> fillingDownCurrent_;
    std::valarray<double> doccCurrent_;
    std::valarray<double> SzCurrent_;

    //values that are accumulated
    std::vector<double> fillingUp_;
    std::vector<double> fillingDown_;
    std::vector<double> docc_;
    std::vector<double> Sz_;

    const size_t N_T_INV_;
}; // class FillingAndDocc

} // namespace Obs
} // namespace Markov
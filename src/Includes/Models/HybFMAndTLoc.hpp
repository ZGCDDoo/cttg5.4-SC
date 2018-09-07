#pragma once

#include "../Utilities/Utilities.hpp"
#include "../Utilities/Integrator.hpp"
#include "../Utilities/Conventions.hpp"

namespace Models
{

template <typename TH0>
class HybFMAndTLoc
{

  public:
    static void CalculateHybFMAndTLoc(const TH0 &h0)
    {
        std::cout << "Start of CalculateHybFMAndTLoc" << std::endl;

        Conventions::MapSS_t mapNames = Conventions::BuildFileNameConventions();
        const std::string tlocFName = mapNames["tlocFile"]; //tloc File Name
        const std::string hybFMFName = mapNames["hybFMFile"];

        using boost::filesystem::exists;
        if ((exists(tlocFName)) && exists(hybFMFName))
        {
            ClusterMatrixCD_t tmp;
            tmp.load(tlocFName);
            if (tmp.n_cols == TH0::Nc)
            {
                return;
            }
        }

        //Get TLoc = Int[t(ktilde)]
        std::cout << "Calculating tLoc" << std::endl;
        TH0 h0_(h0);
        const ClusterMatrixCD_t tlocR = Integrator::CubatureKTilde<TH0>(h0);
        const ClusterMatrixCD_t tlocK = FourierDCA::RtoK(tlocR, h0.RSites(), h0.KWaveVectors());

        //Get Int[ t(ktilde)^2 ]
        std::cout << "Calculating hybFM" << std::endl;
        TKTildeSquared fct(h0_);
        const ClusterMatrixCD_t tktildeSquaredIntegrated = Integrator::CubatureKTilde(fct);

        const ClusterMatrixCD_t hybFMR = tktildeSquaredIntegrated - tlocR * tlocR;
        const ClusterMatrixCD_t hybFMK = FourierDCA::RtoK(hybFMR, h0.RSites(), h0.KWaveVectors());

#ifdef DCA
        tlocK.save(tlocFName, arma::arma_binary);
        hybFMK.save(hybFMFName, arma::arma_binary);
#else
        tlocR.save(tlocFName, arma::arma_binary);
        hybFMR.save(hybFMFName, arma::arma_binary);
#endif
        std::cout << "End of CalculateHybFMAndTLoc" << std::endl;
    }

    struct TKTildeSquared
    {

        TKTildeSquared(const TH0 &h0) : h0_(h0){};
        const size_t n_rows = TH0::n_rows;
        const size_t n_cols = TH0::n_cols;
        const size_t Nx = TH0::Nx;
        const size_t Ny = TH0::Ny;
        const size_t Nc = TH0::Nc;

        TH0 h0_;

        ClusterMatrixCD_t operator()(const double &kTildeX, const double &kTildeY) //return t(ktilde)^2
        {
            const ClusterMatrixCD_t tmp = h0_(kTildeX, kTildeY);
            return (tmp * tmp);
        }
    };
};

} // namespace Models

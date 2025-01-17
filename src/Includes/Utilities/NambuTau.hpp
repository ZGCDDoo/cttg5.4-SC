#pragma once

#include "NambuMat.hpp"
#include "Fourier.hpp"
#include "Utilities.hpp"
#include "MPIUtilities.hpp"
#include "IO.hpp"

namespace NambuTau
{

using namespace NambuMat;
using Data_t = ClusterCube_t;

template <typename TIOModel>
class NambuCluster0Tau
{
    //definit par la fct hyb, tloc, mu et beta et un Nombre de slice de temps NTau

  public:
    const double EPS = 1e-13;
    const double deltaTau = 0.008;

    NambuCluster0Tau() : nambuMatCluster_(), beta_(), NTau_(){};

    NambuCluster0Tau(const NambuCluster0Mat &nambuMatCluster, const size_t &NTau) : ioModel_(),
                                                                                    nambuMatCluster_(nambuMatCluster),
                                                                                    beta_(nambuMatCluster.beta()),
                                                                                    NTau_()
    {
        mpiUt::Print("Creating gtau very inefficiently !!!!! ");

        NTau_ = std::max<double>(static_cast<double>(NTau), beta_ / deltaTau);
        const size_t n_rows = nambuMatCluster_.n_rows();
        const size_t n_cols = nambuMatCluster_.n_cols();
        data_.resize(n_rows, n_cols, NTau_ + 1);

        for (size_t tt = 0; tt < NTau_ + 1; tt++)
        {
            Tau_t tau = beta_ * (static_cast<double>(tt)) / static_cast<double>(NTau_);
            if (tt == 0)
            {
                tau += EPS;
            }
            if (tt == NTau_)
            {
                tau -= EPS;
            }

            data_.slice(tt) = Fourier::MatToTauCluster(nambuMatCluster_, tau);
        }

        if (mpiUt::Rank() == mpiUt::master)
        {
            Save("gtau.dat");
        }

        nambuMatCluster_.Clear();
        mpiUt::Print("gtau Created");
    };

    ~NambuCluster0Tau() = default;

    NambuCluster0Mat nambuMatCluster() const { return nambuMatCluster_; };
    size_t NTau() const { return NTau_; };

    void Clear()
    {
        data_.clear();
        nambuMatCluster_.Clear();
    }

    double operator()(const Site_t &s1, const Site_t &s2, const Tau_t &tauIn, const std::pair<FermionSpin_t, FermionSpin_t> nambuIndices = {FermionSpin_t::Up, FermionSpin_t::Up})
    {
        double tau = tauIn - EPS;

        double aps = 1.0;

        if (tau < 0.0)
        {
            tau += beta_;
            aps = -1.0;
        }

        const double nt = std::abs(tau) / beta_ * static_cast<double>(NTau_);
        const size_t n0 = static_cast<size_t>(nt);
        const size_t r1 = s1 + static_cast<size_t>(nambuIndices.first) * ioModel_.Nc;
        const size_t r2 = s2 + static_cast<size_t>(nambuIndices.second) * ioModel_.Nc;
        // std::cout << "data_.n_rows = " << data_.n_rows << std::endl;
        const double greentau0 = aps * ((1.0 - (nt - n0)) * data_(r1, r2, n0) + (nt - n0) * data_(r1, r2, n0 + 1));
        return greentau0;
    }

    const NambuCluster0Tau &operator=(const NambuCluster0Tau &gf)
    {
        if (this == &gf)
            return *this; //évite les boucles infinies
        nambuMatCluster_ = gf.nambuMatCluster_;
        NTau_ = gf.NTau_;
        beta_ = gf.beta_;
        data_ = gf.data_;
        return *this;
    }

    void Save(const std::string &fileName)
    {

        //to shut-up warning
        std::string temp = fileName;
        temp += "k";

        std::cout << "Nope, not implemented yet save gtau !" << std::endl;
    }

  private:
    TIOModel ioModel_;
    NambuCluster0Mat nambuMatCluster_;
    Data_t data_;
    double beta_;
    size_t NTau_;
};
} // namespace NambuTau

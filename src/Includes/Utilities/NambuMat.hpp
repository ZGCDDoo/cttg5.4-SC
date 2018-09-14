#pragma once
#include "Utilities.hpp"
#include "Fourier_DCA.hpp"

namespace NambuMat
{

class HybridizationMat
{
  public:
    HybridizationMat() : data_(), fm_(), sm_(){};
    HybridizationMat(const ClusterCubeCD_t &data, const ClusterMatrixCD_t &fm, const ClusterMatrixCD_t &sm = ClusterMatrixCD_t()) : data_(data)

    {

        using arma::kron;
        const ClusterMatrixCD_t II2x2 = ClusterMatrixCD_t(2, 2).eye();

        fm_ = kron(II2x2, fm);

        if (sm_.n_rows == 0)
        {
            sm_ = ClusterMatrixCD_t(fm_.n_rows, fm_.n_cols).zeros();
        }
        else
        {
            sm_ = kron(II2x2, sm);
        }

        assert(fm_.n_rows == sm_.n_cols);
        assert(fm_.n_cols == data_.n_rows);
    };

    HybridizationMat(const HybridizationMat &hyb) : data_(hyb.data_), fm_(hyb.fm_), sm_(hyb.sm_){};
    ~HybridizationMat() = default;

    //definit par les moments, et le data qui est determine de facon auto-coherente.
    ClusterMatrixCD_t fm() const { return fm_; };
    ClusterMatrixCD_t sm() const { return sm_; };
    ClusterCubeCD_t data() const { return data_; };

    void Clear()
    {
        data_.clear();
        fm_.clear();
        sm_.clear();
    }

    size_t n_slices() const
    {
        return data_.n_slices;
    }

    ClusterMatrixCD_t slice(const size_t &n)
    {
        return data_.slice(n);
    }

    const HybridizationMat &operator=(const HybridizationMat &hyb)
    {
        if (this == &hyb)
            return *this; //évite les boucles infinies
        fm_ = hyb.fm_;
        sm_ = hyb.sm_;
        data_ = hyb.data_;

        return *this;
    }

    void PatchHF(const size_t &NN, const double &beta)
    {
        const size_t nrows = data_.n_rows;
        const size_t NNBefore = n_slices();
        data_.resize(nrows, nrows, NN);

        for (size_t nn = NNBefore; nn < NN; nn++)
        {
            // std::cout << "Patching !" << std::endl;
            cd_t iwn = cd_t(0.0, (2.0 * nn + 1.0) * M_PI / beta);
            data_.slice(nn) = fm_ / iwn + sm_ / (iwn * iwn);
        }
        // std::cout << "End of patching" << std::endl;
    }

  private:
    ClusterCubeCD_t data_;
    ClusterMatrixCD_t fm_;
    ClusterMatrixCD_t sm_;
};

class NambuCluster0Mat
{
    //definit par la fct hyb, tloc, mu et beta

  public:
    NambuCluster0Mat() : hyb_(),
                         data_(),
                         zm_(), fm_(), sm_(), tm_(),
                         tLoc_(),
                         mu_(),
                         beta_(){};

    NambuCluster0Mat(const NambuCluster0Mat &gf) : hyb_(gf.hyb_),
                                                   data_(gf.data_),
                                                   zm_(gf.zm_), fm_(gf.fm_), sm_(gf.sm_), tm_(gf.tm_),
                                                   tLoc_(gf.tLoc_),
                                                   mu_(gf.mu_),
                                                   beta_(gf.beta_){};

    NambuCluster0Mat(const HybridizationMat &hyb, const ClusterMatrixCD_t &tLoc, const double &mu, const double &beta) : hyb_(hyb),
                                                                                                                         data_(),
                                                                                                                         tLoc_(tLoc),
                                                                                                                         mu_(mu),
                                                                                                                         beta_(beta)
    {
        assert(2 * tLoc_.n_rows == hyb.data().n_rows);
        const size_t NNambu = hyb_.data().n_rows;
        const size_t ll = hyb.data().n_slices;

        data_.resize(NNambu, NNambu, ll);
        data_.zeros();

        //For now, not really nambu, but simply not split in spin
        const ClusterMatrixCD_t II = ClusterMatrixCD_t(NNambu, NNambu).eye();
        const ClusterMatrixCD_t II2x2 = ClusterMatrixCD_t(2, 2).eye();

        zm_ = ClusterMatrixCD_t(NNambu, NNambu).zeros();
        fm_ = II;
        std::cout << "Here" << std::endl;

        using arma::kron;
        const ClusterMatrixCD_t tlocNambu = arma::kron(II2x2, tLoc_);

        std::cout << "tlocNambu.n_rows = " << tlocNambu.n_rows << std::endl;
        std::cout << "II.n_rows = " << II.n_rows << std::endl;

        const ClusterMatrixCD_t tmpsm = tlocNambu - mu_ * II;
        std::cout << "Here 2" << std::endl;

        sm_ = tmpsm;
        tm_ = tmpsm * tmpsm + hyb_.fm();

        ClusterMatrixCD_t tmp;
        for (size_t n = 0; n < ll; n++)
        {
            const cd_t zz = cd_t(mu_, (2.0 * n + 1.0) * M_PI / beta_);
            tmp = zz * II - tlocNambu - hyb_.slice(n);
            data_.slice(n) = tmp.i();
        }
    }

    void Clear()
    {
        data_.clear();
        zm_.clear();
        fm_.clear();
        sm_.clear();
        tm_.clear();
        tLoc_.clear();
        hyb_.Clear();
    }

    ~NambuCluster0Mat() = default;

    const NambuCluster0Mat &operator=(const NambuCluster0Mat &gf)
    {
        if (this == &gf)
            return *this; //évite les boucles infinies
        data_ = gf.data_;
        zm_ = gf.zm_;
        fm_ = gf.fm_;
        sm_ = gf.sm_;
        tm_ = gf.tm_;
        tLoc_ = gf.tLoc_;
        mu_ = gf.mu_;
        beta_ = gf.beta_;
        hyb_ = gf.hyb_;
        return *this;
    }

    ClusterCubeCD_t data() const { return data_; };
    ClusterMatrixCD_t zm() const { return zm_; };
    ClusterMatrixCD_t fm() const { return fm_; };
    ClusterMatrixCD_t sm() const { return sm_; };
    ClusterMatrixCD_t tm() const { return tm_; };
    SiteVectorCD_t tube(const size_t &s1, const size_t s2) const { return data_.tube(s1, s2); };
    double beta() const { return beta_; };
    size_t n_rows() const { return data_.n_rows; };
    size_t n_cols() const { return data_.n_cols; };
    size_t n_slices() const { return data_.n_slices; };

  private:
    HybridizationMat hyb_;

    ClusterCubeCD_t data_;
    ClusterMatrixCD_t zm_;
    ClusterMatrixCD_t fm_;
    ClusterMatrixCD_t sm_;
    ClusterMatrixCD_t tm_;

    ClusterMatrixCD_t tLoc_;

    double mu_;
    double beta_;
};
} // namespace NambuMat
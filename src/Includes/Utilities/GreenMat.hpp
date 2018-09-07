#pragma once
#include "Utilities.hpp"
#include "Fourier_DCA.hpp"

namespace GreenMat
{

class HybridizationMat
{
  public:
    HybridizationMat() : data_(), fm_(){};
    HybridizationMat(const ClusterCubeCD_t &data, const ClusterMatrixCD_t &fm) : data_(data), fm_(fm){};
    HybridizationMat(const HybridizationMat &hyb) : data_(hyb.data_), fm_(hyb.fm_){};
    ~HybridizationMat()
    {
        //delete data_;
        //delete fm_;
    }
    //definit par les moments, et le data qui est determine de facon auto-coherente.
    ClusterMatrixCD_t fm() const { return fm_; };
    ClusterCubeCD_t data() const { return data_; };

    void clear()
    {
        data_.clear();
        fm_.clear();
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
        data_ = hyb.data_;
        fm_ = hyb.fm_;
        return *this;
    }

    void PatchHF(const size_t &NN, const double &beta)
    {
        const size_t nrows = data_.n_rows;
        data_.resize(nrows, nrows, NN);

        for (size_t nn = n_slices(); nn < NN; nn++)
        {
            cd_t iwn = cd_t(0.0, (2.0 * nn + 1.0) * M_PI / beta);
            data_.slice(nn) = fm_ / iwn;
        }
    }

  private:
    ClusterCubeCD_t data_;
    ClusterMatrixCD_t fm_;
};

class GreenCluster0Mat
{
    //definit par la fct hyb, tloc, mu et beta

  public:
    GreenCluster0Mat() : hyb_(),
                         data_(),
                         zm_(), fm_(), sm_(), tm_(),
                         tLoc_(),
                         mu_(),
                         beta_(){};

    GreenCluster0Mat(const GreenCluster0Mat &gf) : hyb_(gf.hyb_),
                                                   data_(gf.data_),
                                                   zm_(gf.zm_), fm_(gf.fm_), sm_(gf.sm_), tm_(gf.tm_),
                                                   tLoc_(gf.tLoc_),
                                                   mu_(gf.mu_),
                                                   beta_(gf.beta_){};

    GreenCluster0Mat(const HybridizationMat &hyb, const ClusterMatrixCD_t &tLoc, const double &mu, const double &beta) : hyb_(hyb),
                                                                                                                         data_(),
                                                                                                                         tLoc_(tLoc),
                                                                                                                         mu_(mu),
                                                                                                                         beta_(beta)
    {
        assert(tLoc_.n_rows == hyb.data().n_rows);
        const size_t Nc = hyb_.data().n_rows;
        const size_t ll = hyb.data().n_slices;

        data_.resize(Nc, Nc, ll);
        data_.zeros();
        zm_.resize(Nc, Nc);
        fm_.resize(Nc, Nc);
        sm_.resize(Nc, Nc);
        tm_.resize(Nc, Nc);

        zm_ = ClusterMatrixCD_t(Nc, Nc).zeros();
        fm_ = ClusterMatrixCD_t(Nc, Nc).eye();
        sm_ = tLoc_ - mu_ * ClusterMatrixCD_t(Nc, Nc).eye();
        tm_ = (tLoc_ - mu_ * ClusterMatrixCD_t(Nc, Nc).eye()) * (tLoc_ - mu_ * ClusterMatrixCD_t(Nc, Nc).eye()) + hyb_.fm();

        ClusterMatrixCD_t tmp;
        for (size_t n = 0; n < ll; n++)
        {
            const cd_t zz = cd_t(mu_, (2.0 * n + 1.0) * M_PI / beta_);
            tmp = zz * ClusterMatrixCD_t(Nc, Nc).eye() - tLoc_ - hyb_.slice(n);
            data_.slice(n) = tmp.i();
        }
    }

    void clear()
    {
        data_.clear();
        zm_.clear();
        fm_.clear();
        sm_.clear();
        tm_.clear();
        tLoc_.clear();
        hyb_.clear();
    }

    ~GreenCluster0Mat()
    {
    }

    void FourierTransform(const ClusterSites_t &RSites, const ClusterSites_t &KWaveVectors)
    {

        //Watch OUT!!! hyb and tloc not fouriertransformed
        data_ = FourierDCA::KtoR(data_, RSites, KWaveVectors);
        zm_ = FourierDCA::KtoR(zm_, RSites, KWaveVectors);
        fm_ = FourierDCA::KtoR(fm_, RSites, KWaveVectors);
        sm_ = FourierDCA::KtoR(sm_, RSites, KWaveVectors);
        tm_ = FourierDCA::KtoR(tm_, RSites, KWaveVectors);
    }

    const GreenCluster0Mat &operator=(const GreenCluster0Mat &gf)
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
} // namespace GreenMat
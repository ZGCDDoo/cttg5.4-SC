#pragma once

#include <armadillo>
#include <string>
#include <boost/serialization/map.hpp>
#include <boost/serialization/valarray.hpp>
#include <boost/serialization/complex.hpp>
#include <ccomplex>
#include <valarray>

#include "../Utilities/MPIUtilities.hpp"

namespace mpiUt
{
template <typename TIOModel>
class IOResult;
}

namespace Result
{

using ClusterMatrixCD_t = arma::cx_mat;
using cd_t = std::complex<double>;

class ISResult
{
    //greenMatUp is the tabularform
  public:
    ISResult(){};

    ISResult(const std::map<std::string, double> &obsScal, const ClusterMatrixCD_t &greenMatUp,
             const ClusterMatrixCD_t &greenMatDown, const std::vector<double> &fillingUp,
             const std::vector<double> &fillingDown) : obsScal_(obsScal),
                                                       n_rows_(greenMatUp.n_rows),
                                                       n_cols_(greenMatUp.n_cols),
                                                       greenTabUp_(n_rows_ * n_cols_),
#ifdef AFM

                                                       greenTabDown_(n_rows_ * n_cols_),
#endif
                                                       fillingUp_(fillingUp.data(), fillingUp.size()),
                                                       fillingDown_(fillingDown.data(), fillingDown.size())
    {
        assert(greenMatDown.n_rows == n_rows_);
#ifdef AFM
        assert(greenMatDown.n_cols == n_cols_);
#endif
        for (size_t j = 0; j < n_cols_; j++)
        {
            for (size_t i = 0; i < n_rows_; i++)
            {
                greenTabUp_[i + n_rows_ * j] = greenMatUp(i, j);
#ifdef AFM
                greenTabDown_[i + n_rows_ * j] = greenMatDown(i, j);
#endif
            }
        }
    }

  private:
    //From boost::mpi and boost::serialze tutorial
    template <typename TIOModel>
    friend class mpiUt::IOResult;
    friend class boost::serialization::access;
    // When the class Archive corresponds to an output archive, the
    // & operator is defined similar to <<.  Likewise, when the class Archive
    // is a type of input archive the & operator is defined similar to >>.
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        //Just to shutUp boost::serialize error for not using version
        size_t versionTmp = version + 1;
        versionTmp++;
        //-----------end of shut-up---------

        ar &obsScal_;
        ar &n_rows_;
        ar &n_cols_;
        ar &greenTabUp_;
        ar &fillingUp_;
        ar &fillingDown_;
#ifdef AFM
        ar &greenTabDown_;
#endif
    }

    std::map<std::string, double> obsScal_;
    size_t n_rows_; //=NMat
    size_t n_cols_; //=Number of independant green functions of the model at hand.

    std::valarray<std::complex<double>> greenTabUp_; //the green in tabular form
                                                     //corresponding to the independant values, in vector form.
                                                     //col .major ordering
#ifdef AFM
    std::valarray<std::complex<double>> greenTabDown_;
#endif
    std::valarray<double> fillingUp_;
    std::valarray<double> fillingDown_;
};
} // namespace Result

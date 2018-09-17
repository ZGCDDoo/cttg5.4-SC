#pragma once

#include <armadillo>
#include <string>
#include <boost/serialization/map.hpp>
#include <boost/serialization/valarray.hpp>
#include <boost/serialization/complex.hpp>
#include <ccomplex>
#include <valarray>

#include "../Utilities/MPIUtilities.hpp"
#include "../Utilities/Utilities.hpp"

namespace mpiUt
{
template <typename TIOModel>
class IOResult;
}

namespace Result
{

using DataCD_t = std::valarray<cd_t>;
using Data_t = std::valarray<double>;

class ISResult
{
    //greenMatUp is the tabularform
  public:
    ISResult(){};

    ISResult(const std::map<std::string, double> &obsScal, const ClusterCubeCD_t &nambu,
             const std::vector<double> &fillingUp, const std::vector<double> &fillingDown) : obsScal_(obsScal),
                                                                                             n_rows_(nambu.n_rows),
                                                                                             n_cols_(nambu.n_cols),
                                                                                             n_slices_(nambu.n_slices),
                                                                                             nambu_(),
                                                                                             fillingUp_(fillingUp.data(), fillingUp.size()),
                                                                                             fillingDown_(fillingDown.data(), fillingDown.size())
    {
        nambu_ = Utilities::CubeCDToVecCD<DataCD_t>(nambu);

        std::cout << "End of ISResult constructor " << std::endl;
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
        ar &n_slices_;
        ar &nambu_;
        ar &fillingUp_;
        ar &fillingDown_;
    }

    std::map<std::string, double> obsScal_;
    size_t n_rows_;
    size_t n_cols_;
    size_t n_slices_;

    DataCD_t nambu_;
    Data_t fillingUp_;
    Data_t fillingDown_;
};
} // namespace Result

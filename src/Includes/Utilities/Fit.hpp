#pragma once

#include "Utilities.hpp"

namespace Fit
{

template <typename TIOModel>
void NFit(const size_t &iter, const double &fact, const double &beta) //here fact is equal to U
{
    std::string fname = "self" + std::to_string(iter) + std::string(".dat");
    ClusterMatrix_t self; //in .dat format
    self.load(fname);

    double filling = 0.0;
    TIOModel ioModel;
    const double CUTOFF = (2.0 * self.n_rows + 1.0) * M_PI / beta;

    for (size_t ii = 0; ii < ioModel.indepSites().size(); ii++)
    {
        if (ioModel.indepSites().at(ii).first == ioModel.indepSites().at(ii).second)
        {

            size_t nelems = CUTOFF < 90 ? 10 : self.n_rows - 0.5 * (60.0 * beta / M_PI - 1.0);
            std::cout << "nelems, cutoff = " << nelems << " " << CUTOFF << std::endl;
            double self0 = arma::sum(self.col(1 + 2 * ii).tail(nelems)) / double(nelems);
            std::cout << "self0 = " << self0 << std::endl;

            filling += 2.0 * self0 * double(ioModel.nOfAssociatedSites().at(ii)) / (double(ioModel.nOfFillingSites()) * fact);
        }
    }

    std::string fname_nFit = std::string("nFit.dat");
    std::ofstream fout(fname_nFit, std::ios_base::out | std::ios_base::app);
    fout << iter << " " << filling << std::endl;
    fout.close();
}
}

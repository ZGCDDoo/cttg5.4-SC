#pragma once

#include "../Models/ModelSquare2x2_AFM.hpp"
#include "../Models/ModelTriangle2x2.hpp"
#include "Utilities.hpp"
#include "IO.hpp"
#include "Conventions.hpp"

namespace SymmetrizeSpins
{

//This should simply be get downEquivalentSites

std::vector<size_t> GetDownEquivalentSites(const std::string &modelType)
{

    if (modelType == "Square2x2_AFM")
    {

        return IO::IOSquare2x2_AFM().downEquivalentSites();
    }
    else if (modelType == "Triangle2x2_AFM")
    {
        return IO::IOTriangle2x2().downEquivalentSites();
    }

    throw std::runtime_error("Miseria: modelType error in params file. Stupido !");
}

void SymmetrizeUpAndDown(const Json &jj)
{

    //Here we read the greens.dat files as specified in conventions.hpp and average them accordingly to the symmetries of up and down spins
    // remembert that the 0th column is the frequencies and that we have the real and imaginary parts in adjacent columns.

    const std::string modelType = jj["modelType"].get<std::string>();
    const std::vector<size_t> downEquivalentSites = GetDownEquivalentSites(modelType);

    if (downEquivalentSites.empty())
    {
        return;
    }

    const Conventions::NameVector_t nameVecGreens = Conventions::BuildGreensVectorNames();

    for (size_t ii = 0; ii < nameVecGreens.size() / 2; ii++)
    {
        std::cout << "ii = " << ii << ", " << nameVecGreens.at(2 * ii) << ", " << nameVecGreens.at(2 * ii + 1) << std::endl;

        const std::string fNameUp = nameVecGreens.at(2 * ii);
        ClusterMatrix_t tmpUp;
        assert(tmpUp.load(fNameUp));

        const std::string fNameDown = nameVecGreens.at(2 * ii + 1);
        ClusterMatrix_t tmpDown;
        assert(tmpDown.load(fNameDown));

        assert(tmpDown.n_cols == tmpUp.n_cols);
        assert(tmpUp.n_cols == (1 + 2 * downEquivalentSites.size()));

        //Remember, the first column is the matsubara axis, so start at 1 for the column index
        for (size_t jj = 0; jj < downEquivalentSites.size(); jj++)
        {
            const size_t upIndex = 1 + 2 * downEquivalentSites.at(jj);
            const size_t downIndex = 1 + 2 * jj;
            tmpUp.col(upIndex) += tmpDown.col(downIndex);
            tmpUp.col(upIndex + 1) += tmpDown.col(downIndex + 1);

            std::cout << std::endl;
            std::cout << "upIndex, DownIndex = " << upIndex << ", " << downIndex << std::endl;
            std::cout << "upIndex + 1, DownIndex + 1 = " << upIndex + 1 << ", " << downIndex + 1 << std::endl;
            std::cout << std::endl;

            tmpUp.col(upIndex) /= 2.0;
            tmpUp.col(upIndex + 1) /= 2.0;

            tmpDown.col(downIndex) = tmpUp.col(upIndex);
            tmpDown.col(downIndex + 1) = tmpUp.col(upIndex + 1);
        }

        assert(tmpUp.save(fNameUp, arma::raw_ascii));
        assert(tmpDown.save(fNameDown, arma::raw_ascii));
    }

    mpiUt::Print("After symmetrize Spins.");
}

} // namespace SymmetrizeSpins

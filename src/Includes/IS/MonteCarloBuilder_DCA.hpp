#pragma once

#include "MonteCarlo.hpp"

#include "MarkovChain.hpp"
#include "MarkovChainAux.hpp"
#include "../Models/SIAM_Square.hpp"
#include "../Models/ModelSquare4x4_DCA.hpp"
#include "../Models/ModelSquare2x2.hpp"
#include "../Utilities/MPIUtilities.hpp"

namespace MC
{

std::unique_ptr<ABC_MonteCarlo> MonteCarloBuilder(const Json &jj, const size_t &seed)
{
#ifdef HAVEMPI
    mpi::environment env;
    mpi::communicator world;
#endif

    const std::string modelType = jj["modelType"].get<std::string>();
    const std::string solverType = jj["solver"].get<std::string>();
    const std::string Int = "Int";
    const std::string Aux = "Aux";

    if (modelType == "SIAM_Square")
    {
        using Model_t = Models::SIAM_Square;
        using IOModel_t = IO::IOSIAM;
        using MarkovInt_t = Markov::MarkovChain<IOModel_t, Model_t>;
        using MarkovAux_t = Markov::MarkovChainAux<IOModel_t, Model_t>;
        //Init a dummy model just to be sure that all files are present:
        if (mpiUt::Rank() == mpiUt::master)
        {
            const Model_t modelDummy(jj);
        }
#ifdef HAVEMPI
        world.barrier();
#endif

        if (solverType == Int)
        {
            return std::make_unique<MC::MonteCarlo<MarkovInt_t>>(std::make_shared<MarkovInt_t>(jj, seed), jj);
        }
        else if (solverType == Aux)
        {
            return std::make_unique<MC::MonteCarlo<MarkovAux_t>>(std::make_shared<MarkovAux_t>(jj, seed), jj);
        }
    }
    else if (modelType == "Square2x2_DCA")
    {
        using Model_t = Models::ModelSquare2x2;
        using IOModel_t = IO::IOSquare2x2;

        using MarkovInt_t = Markov::MarkovChain<IOModel_t, Model_t>;
        using MarkovAux_t = Markov::MarkovChainAux<IOModel_t, Model_t>;
        //Init a dummy model just to be sure that all files are present:
        if (mpiUt::Rank() == mpiUt::master)
        {
            const Model_t modelDummy(jj);
        }
#ifdef HAVEMPI
        world.barrier();
#endif

        if (solverType == Int)
        {
            return std::make_unique<MC::MonteCarlo<MarkovInt_t>>(std::make_shared<MarkovInt_t>(jj, seed), jj);
        }
        else if (solverType == Aux)
        {
            return std::make_unique<MC::MonteCarlo<MarkovAux_t>>(std::make_shared<MarkovAux_t>(jj, seed), jj);
        }
    }
    else if (modelType == "Square4x4_DCA")
    {
        using Model_t = Models::ModelSquare4x4_DCA;
        using IOModel_t = IO::IOSquare4x4_DCA;

        using MarkovInt_t = Markov::MarkovChain<IOModel_t, Model_t>;
        using MarkovAux_t = Markov::MarkovChainAux<IOModel_t, Model_t>;
        //Init a dummy model just to be sure that all files are present:
        if (mpiUt::Rank() == mpiUt::master)
        {
            const Model_t modelDummy(jj);
        }
#ifdef HAVEMPI
        world.barrier();
#endif

        if (solverType == Int)
        {
            return std::make_unique<MC::MonteCarlo<MarkovInt_t>>(std::make_shared<MarkovInt_t>(jj, seed), jj);
        }
        else if (solverType == Aux)
        {
            return std::make_unique<MC::MonteCarlo<MarkovAux_t>>(std::make_shared<MarkovAux_t>(jj, seed), jj);
        }
    }

    throw std::runtime_error("Miseria: solver and or modelType error in params file. Stupido !");

    return NULL;
}

} // namespace MC

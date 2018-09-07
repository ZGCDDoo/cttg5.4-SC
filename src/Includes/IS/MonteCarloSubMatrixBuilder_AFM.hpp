#pragma once

#include "MonteCarlo.hpp"

#include "MarkovChainSubMatrix.hpp"
#include "MarkovChainAuxSubMatrix.hpp"
#include "../Models/ModelSquare2x2_AFM.hpp"
#include "../Models/ModelTriangle2x2.hpp"
#include "../Utilities/MPIUtilities.hpp"

namespace MC
{

std::unique_ptr<ABC_MonteCarlo> MonteCarloBuilder_AFM(const Json &jj, const size_t &seed)
{
#ifdef HAVEMPI
    mpi::environment env;
    mpi::communicator world;
#endif

    const std::string modelType = jj["modelType"].get<std::string>();
    const std::string solverType = jj["solver"].get<std::string>();
    const std::string IntSub = "IntSub";
    const std::string AuxSub = "AuxSub";

    if (modelType == "Square2x2_AFM")
    {
        using Model_t = Models::ModelSquare2x2_AFM;
        using IOModel_t = IO::IOSquare2x2_AFM;
        using MarkovIntSub_t = Markov::MarkovChainSub<IOModel_t, Model_t>;
        using MarkovAuxSub_t = Markov::MarkovChainAuxSub<IOModel_t, Model_t>;
        //Init a dummy model just to be sure that all files are present:
        if (mpiUt::Rank() == mpiUt::master)
        {
            const Model_t modelDummy(jj);
        }
#ifdef HAVEMPI
        world.barrier();
#endif

        if (solverType == IntSub)
        {
            return std::make_unique<MC::MonteCarlo<MarkovIntSub_t>>(std::make_shared<MarkovIntSub_t>(jj, seed), jj);
        }
        else if (solverType == AuxSub)
        {
            return std::make_unique<MC::MonteCarlo<MarkovAuxSub_t>>(std::make_shared<MarkovAuxSub_t>(jj, seed), jj);
        }
    }
    else if (modelType == "Triangle2x2_AFM")
    {
        using Model_t = Models::ModelTriangle2x2;
        using IOModel_t = IO::IOTriangle2x2;
        using MarkovIntSub_t = Markov::MarkovChainSub<IOModel_t, Model_t>;
        using MarkovAuxSub_t = Markov::MarkovChainAuxSub<IOModel_t, Model_t>;
        //Init a dummy model just to be sure that all files are present:
        if (mpiUt::Rank() == mpiUt::master)
        {
            const Model_t modelDummy(jj);
        }
#ifdef HAVEMPI
        world.barrier();
#endif

        if (solverType == IntSub)
        {
            return std::make_unique<MC::MonteCarlo<MarkovIntSub_t>>(std::make_shared<MarkovIntSub_t>(jj, seed), jj);
        }
        else if (solverType == AuxSub)
        {
            return std::make_unique<MC::MonteCarlo<MarkovAuxSub_t>>(std::make_shared<MarkovAuxSub_t>(jj, seed), jj);
        }
    }

    throw std::runtime_error("Miseria: solver and or modelType error in params file. Stupido !");

    return NULL;
}

} // namespace MC

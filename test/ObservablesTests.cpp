#define SUBMATRIX

#include <gtest/gtest.h>

#include "../src/Includes/IS/Obs/Observables.hpp"
#include "../src/Includes/Models/ModelSquare2x2.hpp"

using Model_t = Models::ModelSquare2x2;
using IOModel_t = IO::IOSquare2x2;
using GreenBinning_t = Markov::Obs::GreenBinning<IO::IOSquare2x2, Models::ModelSquare2x2>;
using Obs_t = Markov::Obs::Observables<IO::IOSquare2x2, Models::ModelSquare2x2>;
using ISDataCT_t = Markov::Obs::ISDataCT<IO::IOSquare2x2, Models::ModelSquare2x2>;

// const double DELTA = 1e-11;
const std::string FNAME = "../test/data/cdmft_square2x2/params1.json";

Obs_t BuildObs() //for Square2x2
{

    std::ifstream fin(FNAME);
    Json jj;
    fin >> jj;
    fin.close();
    Model_t model(jj);
    std::shared_ptr<ISDataCT_t> dataCT(
        new ISDataCT_t(
            jj["beta"].get<double>(),
            model, jj["NTAU"].get<double>()));

    std::shared_ptr<Model_t> modelPtr(new Model_t(jj));

    Obs_t obs(dataCT, jj);
    return obs;
}

TEST(ObsTests, Init)
{
    Obs_t Obs = BuildObs();
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

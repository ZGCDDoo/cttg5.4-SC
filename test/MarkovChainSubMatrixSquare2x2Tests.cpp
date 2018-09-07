#define SUBMATRIX
#include <gtest/gtest.h>

#include "../src/Includes/IS/MarkovChainSubMatrix.hpp"
#include "../src/Includes/Models/ModelSquare2x2.hpp"

using namespace LinAlg;

const size_t Nx = 2;

using Model_t = Models::ModelSquare2x2;
using IOModel_t = IO::IOSquare2x2;
using H0_t = Models::H0Square<Nx, Nx>;
using Markov_t = Markov::MarkovChainSub<IOModel_t, Model_t>;

const double DELTA = 1e-11;
const std::string FNAME = "../test/data/cdmft_square2x2/params1.json";
typedef LinAlg::Matrix_t Matrix_t;

Markov_t BuildMarkovChain()
{
    std::ifstream fin(FNAME);
    Json jj;
    fin >> jj;
    fin.close();
    std::cout << "Reading in Json in BuildMarkovChain() " << std::endl;
    const size_t seed = 10224;
    jj["KMAX_UPD"] = 50;
    jj["UPDATESMEAS"] = 20;
    Markov_t markovchain(jj, seed);
    std::cout << "After BuildMarkovChain() " << std::endl;
    return markovchain;
}

TEST(MarkovChainSquare2x2Tests, Init)
{
    Markov_t mc = BuildMarkovChain();
}

TEST(MarkovChainSquare2x2Tests, DoStep)
{
    Markov_t mc = BuildMarkovChain();

    for (size_t ii = 0; ii < 2; ii++)
    {
        mc.DoStep();
    }
    mc.CleanUpdate();

    for (size_t ii = 0; ii < 1000; ii++)
    {
        mc.DoStep();
    }

    mc.CleanUpdate();
    for (size_t ii = 0; ii < 1000; ii++)
    {
        mc.DoStep();
    }

    std::cout << "After DOstep " << std::endl;

    Matrix_t tmpUp;
    Matrix_t tmpDown;
    tmpUp = mc.Nup();
    tmpDown = mc.Ndown();
    std::cout << "Mup.size = " << tmpUp.n_cols() << std::endl;
    mc.CleanUpdate();

    for (size_t i = 0; i < tmpUp.n_rows(); i++)
    {
        for (size_t j = 0; j < tmpUp.n_rows(); j++)
        {
            ASSERT_NEAR(tmpUp(i, j), mc.Nup()(i, j), DELTA);
            ASSERT_NEAR(tmpDown(i, j), mc.Ndown()(i, j), DELTA);
        }
    }

    ASSERT_EQ(tmpUp.n_cols(), mc.Nup().n_rows());
    ASSERT_EQ(tmpUp.n_cols(), tmpUp.n_rows());
    std::cout << "dims = " << tmpUp.n_cols() << std::endl;
    mc.SaveTherm();
    // mc.Nup().Print();
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

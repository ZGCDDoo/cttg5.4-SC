#include <gtest/gtest.h>

#include "../src/Includes/IS/MarkovChain.hpp"
#include "../src/Includes/Models/SIAM_Square.hpp"

using namespace LinAlg;

const double DELTA = 3e-12;
const std::string FNAME = "../test/data/DMFT/test_dmft0.json";

Markov::MarkovChain<IO::IOSIAM, Models::SIAM_Square> BuildMarkovChain() //for SIAM_Square
{
    std::ifstream fin(FNAME);
    Json jj;
    fin >> jj;
    fin.close();
    std::cout << "Reading in Json in BuildMarkovChain() " << std::endl;
    const size_t seed = 10224;
    Markov::MarkovChain<IO::IOSIAM, Models::SIAM_Square> markovchain(jj, seed);
    std::cout << "After BuildMarkovChain() " << std::endl;
    return markovchain;
}

TEST(MarkovChainTests, Init)
{
    Markov::MarkovChain<IO::IOSIAM, Models::SIAM_Square> mc = BuildMarkovChain();
}

TEST(MonteCarloTest, DoStep)
{
    Markov::MarkovChain<IO::IOSIAM, Models::SIAM_Square> mc = BuildMarkovChain();

    for (size_t i = 0; i < 50; i++)
    {
        mc.InsertVertex();
    }

    std::cout << "After Insert " << std::endl;
    for (size_t i = 0; i < 11; i++)
    {
        mc.InsertVertex();
        mc.RemoveVertex();
    }

    std::cout << "After Remove " << std::endl;
    size_t ii = 0;
    for (ii = 0; ii < 10000; ii++)
    {
        mc.DoStep();
    }
    std::cout << "ii = " << ii << std::endl;

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
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

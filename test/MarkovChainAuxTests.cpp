#include <gtest/gtest.h>

#include "../src/Includes/IS/MarkovChainAux.hpp"
#include "../src/Includes/Models/SIAM_Square.hpp"

const double DELTA = 3e-12;
const std::string FNAME = "../test/data/DMFT/test_dmft0.json";
using LinAlg::Matrix_t;

Markov::MarkovChainAux<IO::IOSIAM, Models::SIAM_Square> BuildMarkovChain() //for SIAM_Square
{
    std::ifstream fin(FNAME);
    Json jj;
    fin >> jj;
    fin.close();
    std::cout << "Reading in Json in BuildMarkovChain() " << std::endl;
    const size_t seed = 10224;
    Markov::MarkovChainAux<IO::IOSIAM, Models::SIAM_Square> markovchain(jj, seed);
    std::cout << "After BuildMarkovChain() " << std::endl;
    return markovchain;
}

TEST(MarkovChainTests, Init)
{
    Markov::MarkovChainAux<IO::IOSIAM, Models::SIAM_Square> mc = BuildMarkovChain();
}

TEST(MarkovChainTests, InsertVertex)
{
    Markov::MarkovChainAux<IO::IOSIAM, Models::SIAM_Square> mc = BuildMarkovChain();

    while (mc.vertices().size() == 0)
    {
        mc.InsertVertex();
    }

    //TODO
    // Vertex vertex0 = mc.vertices()[0];
    // double Nup = 1.0 / (mc.GetGreenTau0Up(vertex0, vertex0) - mc.model().auxUp(vertex0));
    // double Ndown = 1.0 / (mc.GetGreenTau0Up(vertex0, vertex0) - mc.model().auxDown(vertex0));
    // ASSERT_DOUBLE_EQ(Nup, mc.Nup()(0, 0));
    // ASSERT_DOUBLE_EQ(Ndown, mc.Ndown()(0, 0));

    for (size_t i = 0; i < 51; i++)
    {
        mc.InsertVertex();
    }

    Matrix_t tmpUp = mc.Nup();
    Matrix_t tmpDown = mc.Ndown();
    mc.CleanUpdate();

    for (size_t i = 0; i < tmpUp.n_rows(); i++)
    {
        for (size_t j = 0; j < tmpUp.n_rows(); j++)
        {
            ASSERT_NEAR(tmpUp(i, j), mc.Nup()(i, j), DELTA);
            ASSERT_NEAR(tmpDown(i, j), mc.Ndown()(i, j), DELTA);
        }
    }
}

TEST(MonteCarloTest, DoStep)
{
    Markov::MarkovChainAux<IO::IOSIAM, Models::SIAM_Square> mc = BuildMarkovChain();

    for (size_t i = 0; i < 50; i++)
    {
        mc.InsertVertex();
    }

    std::cout << "After Insert " << std::endl;
    for (size_t i = 0; i < 11; i++)
    {
        mc.RemoveVertex();
    }

    std::cout << "After Remove " << std::endl;
    for (size_t i = 0; i < 10000; i++)
    {
        mc.DoStep();
    }

    std::cout << "After DOstep " << std::endl;

    Matrix_t tmpUp;
    Matrix_t tmpDown;
    tmpUp = mc.Nup();
    tmpDown = mc.Ndown();
    std::cout << "Nup.size = " << tmpUp.n_cols() << std::endl;
    mc.CleanUpdate();

    for (size_t i = 0; i < tmpUp.n_rows(); i++)
    {
        for (size_t j = 0; j < tmpUp.n_rows(); j++)
        {
            ASSERT_NEAR(tmpUp(i, j), mc.Nup()(i, j), DELTA);
            ASSERT_NEAR(tmpUp(i, j), mc.Nup()(i, j), DELTA);
            ASSERT_NEAR(tmpDown(i, j), mc.Ndown()(i, j), DELTA);
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

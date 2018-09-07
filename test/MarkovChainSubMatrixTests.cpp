#define SUBMATRIX

#include <gtest/gtest.h>

#include "../src/Includes/IS/MarkovChainSubMatrix.hpp"
#include "../src/Includes/Models/SIAM_Square.hpp"

const double DELTA = 1e-11;
const std::string FNAME = "../test/data/DMFT/test_dmft0.json";
typedef LinAlg::Matrix_t Matrix_t;

Markov::MarkovChainSub<IO::IOSIAM, Models::SIAM_Square> BuildMarkovChain() //for SIAM_Square
{
    std::ifstream fin(FNAME);
    Json jj;
    fin >> jj;
    fin.close();
    std::cout << "Reading in Json in BuildMarkovChain() " << std::endl;
    const size_t seed = 10224;
    Markov::MarkovChainSub<IO::IOSIAM, Models::SIAM_Square> markovchain(jj, seed);
    std::cout << "After BuildMarkovChain() " << std::endl;
    return markovchain;
}

// TEST(MarkovChainTests, Init)
// {
//     Markov::MarkovChainIntSub<IO::IOSIAM, Models::SIAM_Square> mc = BuildMarkovChain();
// }

TEST(MonteCarloTest, DoStep)
{
    Markov::MarkovChainSub<IO::IOSIAM, Models::SIAM_Square> mc = BuildMarkovChain();

    //mc.CleanUpdate();

    for (size_t i = 0; i < 2000; i++)
    {
        mc.DoStep();
        // mc.InsertVertex();
        // mc.PreparationSteps();
        // mc.RemoveVertexSubmatrix();
        // mc.InsertVertexSubmatrix();
        // mc.RemoveVertexSubmatrix();
        // mc.InsertVertexSubmatrix();
        // mc.UpdateSteps();
    }
    std::cout << "After DOstep " << std::endl;

    Matrix_t tmpUp;
    Matrix_t tmpDown;
    tmpUp = mc.Nup();
    tmpDown = mc.Ndown();
    std::cout << "Mup.size = " << tmpUp.n_cols() << std::endl;
    mc.CleanUpdate();
    std::cout << "Nup after cleanupdate = " << std::endl;
    // mc.Nup().Print();

    // std::cout << "diff Nup - Nup = " << std::endl;
    // (tmpUp - mc.Nup()).print();

    for (size_t i = 0; i < tmpUp.n_rows(); i++)
    {
        for (size_t j = 0; j < tmpUp.n_cols(); j++)
        {
            double tmp = std::abs(tmpUp(i, j) - mc.Nup()(i, j));
            if (tmp > DELTA)
            {
                std::cout << "Nup not close, value = " << tmp << std::endl;
            }
            tmp = std::abs(tmpDown(i, j) - mc.Ndown()(i, j));
            if (tmp > DELTA)
            {
                std::cout << "Ndown not close, value = " << tmp << std::endl;
            }
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

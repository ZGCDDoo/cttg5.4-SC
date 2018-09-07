

#include <gtest/gtest.h>
#include "../src/Includes/Utilities/Integrator.hpp"
#include "../src/Includes/Models/H0Triangle.hpp"

using namespace Integrator;

const double DELTA = 1.49e-8;

struct FctTest
{
  public:
    static const size_t Nc = 4;
    static const size_t TNX = 2;
    static const size_t TNY = 2;

    FctTest(){};

    const size_t n_rows = Nc;
    const size_t n_cols = Nc;

    ClusterMatrixCD_t operator()(const double &x, const double &y)
    {
        ClusterMatrixCD_t tmp(Nc, Nc);
        tmp.zeros();
        for (size_t i = 0; i < Nc; i++)
        {
            for (size_t j = 0; j < Nc; j++)
            {
                tmp(i, j) = cd_t(i * x + i * i * x * x, j * y);
            }
        }
        return tmp;
    }

  private:
};

TEST(IntegratorTest, Init)
{

    const double t = -1.0;
    const double tp = -0.4;
    const double tpp = 0.0;

    using H0_t = Models::H0Triangle<2, 2>;
    H0_t h0(t, tp, tpp);
    ClusterMatrixCD_t tLocTest = GridKTilde(h0, 100);
    const ClusterMatrixCD_t goodResult = {{0.0, t, t, tp},
                                          {t, 0.0, 0.0, t},
                                          {t, 0.0, 0.0, t},
                                          {tp, t, t, 0.0}};

    for (size_t i = 0; i < goodResult.n_rows; i++)
    {
        for (size_t j = 0; j < goodResult.n_rows; j++)
        {

            ASSERT_NEAR(tLocTest(i, j).real(), goodResult(i, j).real(), DELTA);
            ASSERT_NEAR(tLocTest(i, j).imag(), goodResult(i, j).imag(), DELTA);
        }
    }

    //====================================
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

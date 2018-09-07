

#include <gtest/gtest.h>

#include "../src/Includes/Utilities/Fourier_DCA.hpp"
#include "../src/Includes/Models/H0Square.hpp"

using namespace FourierDCA;

const double DELTA = 1e-11;
TEST(FourierDCATests, KToR)
{
    arma::arma_rng::set_seed_random();
    const size_t Nx = 3;
    const size_t NMat = 1;
    const Models::H0Square<Nx, Nx> h0(-1.0, -0.32, 0.15);

    DataK_t greenK(Nx * Nx, Nx * Nx, NMat);
    greenK.zeros();
    for (size_t nn = 0; nn < NMat; nn++)
    {
        SiteVectorCD_t tmp(Nx * Nx);
        tmp.randu();
        // tmp(1) = tmp(2);
        greenK.slice(nn).diag() = tmp;
        greenK.slice(nn).print();
        std::cout << "\n\n";
    }

    ClusterCubeCD_t greenRTest = KtoR(greenK, h0.RSites(), h0.KWaveVectors());
    DataK_t greenKTest = RtoK(greenRTest, h0.RSites(), h0.KWaveVectors());

    assert(greenKTest.n_cols == Nx * Nx);
    assert(greenKTest.n_rows == Nx * Nx);
    assert(greenK.n_rows == Nx * Nx);
    assert(greenKTest.n_cols == Nx * Nx);

    std::cout << "\n\n"
              << " greenR " << std::endl;
    greenRTest.print();

    std::cout << "\n\n"
              << " greenKTest " << std::endl;
    greenKTest.print();

    std::cout << "\n\n"
              << " greenK " << std::endl;
    greenK.print();

    std::cout << "Here " << std::endl;
    for (size_t nn = 0; nn < NMat; nn++)
    {
        for (size_t ii = 0; ii < greenKTest.n_rows; ii++)
        {
            for (size_t jj = 0; jj < greenKTest.n_cols; jj++)
            {
                // std::cout << "ii, jj = " << ii << ", " << jj << std::endl;
                ASSERT_NEAR(greenKTest(ii, jj, nn).real(), greenK(ii, jj, nn).real(), DELTA);
                ASSERT_NEAR(greenKTest(ii, jj, nn).imag(), greenK(ii, jj, nn).imag(), DELTA);
            }
        }
    }
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

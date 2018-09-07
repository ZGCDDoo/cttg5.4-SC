

#include <gtest/gtest.h>
#include "../src/Includes/Models/H0Triangle.hpp"
#include "../src/Includes/Utilities/Integrator.hpp"

TEST(H0Triangle2DTest, Init)
{

    Models::H0Triangle<2, 2> h0(-1.0, -0.4, 0.0);

    ASSERT_DOUBLE_EQ(h0.t(), -1.0);
    ASSERT_DOUBLE_EQ(h0.tPrime(), -0.4);
    ASSERT_DOUBLE_EQ(h0.tPrimePrime(), 0.0);
    ASSERT_EQ(h0.RSites().size(), 4);
    ASSERT_EQ(h0.KWaveVectors().size(), 4);
    ASSERT_EQ(h0(0.0, 0.0).size(), 16);

    ASSERT_DOUBLE_EQ(h0.KWaveVectors()[0][0], 0.0);
    ASSERT_DOUBLE_EQ(h0.KWaveVectors()[0][1], 0.0);
    ASSERT_DOUBLE_EQ(h0.KWaveVectors()[1][0], M_PI);
    ASSERT_DOUBLE_EQ(h0.KWaveVectors()[1][1], 0.0);
    ASSERT_DOUBLE_EQ(h0.KWaveVectors()[2][0], 0.0);
    ASSERT_DOUBLE_EQ(h0.KWaveVectors()[2][1], M_PI);
    ASSERT_DOUBLE_EQ(h0.KWaveVectors()[3][0], M_PI);
    ASSERT_DOUBLE_EQ(h0.KWaveVectors()[3][1], M_PI);

    ASSERT_DOUBLE_EQ(h0.RSites()[0][0], 0.0);
    ASSERT_DOUBLE_EQ(h0.RSites()[0][1], 0.0);
    ASSERT_DOUBLE_EQ(h0.RSites()[1][0], 1.0);
    ASSERT_DOUBLE_EQ(h0.RSites()[1][1], 0.0);
    ASSERT_DOUBLE_EQ(h0.RSites()[2][0], 0.0);
    ASSERT_DOUBLE_EQ(h0.RSites()[2][1], 1.0);
    ASSERT_DOUBLE_EQ(h0.RSites()[3][0], 1.0);
    ASSERT_DOUBLE_EQ(h0.RSites()[3][1], 1.0);

    ASSERT_DOUBLE_EQ(-4.6847345710802575, h0.Eps0k(-0.1, 0.3));
}

TEST(H0Triangle2DTest, Hopping)
{

    Models::H0Triangle<2, 2> h0(-1.0, -0.4, 0.0);

    ClusterMatrixCD_t GoodHoppingKTilde = {
        {cd_t(0.0, 0.0), cd_t(-1.98006658, 0.19866933), cd_t(-1.82533561, -0.56464247), cd_t(-0.76842440, -0.15576734)},
        {cd_t(-1.98006658, -0.19866933), cd_t(0.0, 0.0), cd_t(-0.72216088, -0.30532472), cd_t(-1.82533561, -0.56464247)},
        {cd_t(-1.82533561, 0.56464247), cd_t(-0.72216088, 0.30532472), cd_t(0.0, 0.0), cd_t(-1.98006658, 0.19866933)},
        {cd_t(-0.76842440, 0.15576734), cd_t(-1.82533561, 0.56464247), cd_t(-1.98006658, -0.19866933), cd_t(0.0, 0.0)}};

    ClusterMatrixCD_t hopping = h0(0.1, -0.3);
    for (size_t i = 0; i < 4; i++)
        for (size_t j = 0; j < 4; j++)
        {
            //std::cout << "i ,j " << i << " " << j << std::endl;
            ASSERT_NEAR(hopping(i, j).real(), GoodHoppingKTilde(i, j).real(), 1e-7);
            ASSERT_NEAR(hopping(i, j).imag(), GoodHoppingKTilde(i, j).imag(), 1e-7);
        }

    // auto tLoc = Integrator::CubatureKTilde(h0);
    // tLoc.print();
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

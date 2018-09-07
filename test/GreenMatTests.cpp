

#include <gtest/gtest.h>
#include "../src/Includes/Utilities/GreenMat.hpp"

//TO DO MOve greenMat and GreenTau tests to seperate files.

const double DELTA = 1e-5;
const double mu = 2.94;
const double Beta = 10.1;

GreenMat::GreenCluster0Mat BuildGreenMat()
{
    ClusterMatrixCD_t tLoc(4, 4);
    ClusterMatrixCD_t fmhyb(4, 4);
    tLoc.zeros();
    fmhyb.zeros();

    tLoc(0, 1) = tLoc(1, 0) = tLoc(1, 3) = tLoc(3, 1) = tLoc(2, 3) = tLoc(3, 2) = tLoc(0, 2) = tLoc(2, 0) = 0.90;
    const double fm = 2.0 * 0.90; //Not right it should be 2.0*0.9^2, but unconsequential for testing purposes
    fmhyb(0, 0) = fmhyb(1, 1) = fmhyb(2, 2) = fmhyb(3, 3) = fm;
    ClusterCubeCD_t hybdata(4, 4, 2);
    hybdata.zeros();

    for (size_t n = 0; n < hybdata.n_slices; n++)
    {
        cd_t iwn = cd_t(0.0, (2.0 * n + 1) * M_PI / Beta);
        hybdata.slice(n) = fm / iwn * ClusterMatrixCD_t(4, 4).eye();
    }

    GreenMat::HybridizationMat hybMat(hybdata, fmhyb);

    //ClusterCube_t goodGreenMat = {}
    GreenMat::GreenCluster0Mat greenCluster0Mat(hybMat, tLoc, mu, Beta);
    //greenCluster0Mat.data().print();
    return greenCluster0Mat;
}

TEST(GreenMatTest, Init)
{

    //ClusterCube_t goodGreenMat = {}
    GreenMat::GreenCluster0Mat greenCluster0Mat = BuildGreenMat();
    //greenCluster0Mat.data().print();

    ClusterMatrixCD_t goodResult0 = {
        {cd_t(0.05934704, -0.13169938), cd_t(-0.01245951, -0.01405697), cd_t(-0.01245951, -0.01405697), cd_t(-0.00480552, 0.00136094)},
        {cd_t(-0.01245951, -0.01405697), cd_t(0.05934704, -0.13169938), cd_t(-0.00480552, 0.00136094), cd_t(-0.01245951, -0.01405697)},
        {cd_t(-0.01245951, -0.01405697), cd_t(-0.00480552, 0.00136094), cd_t(0.05934704, -0.13169938), cd_t(-0.01245951, -0.01405697)},
        {cd_t(-0.00480552, 0.00136094), cd_t(-0.01245951, -0.01405697), cd_t(-0.01245951, -0.01405697), cd_t(0.05934704, -0.13169938)}};

    ClusterMatrixCD_t goodResult1 = {
        {cd_t(0.15599519, -0.18372943), cd_t(-0.00862298, -0.05205002), cd_t(-0.00862298, -0.05205002), cd_t(-0.01863849, -0.0137227)},
        {cd_t(-0.00862298, -0.05205002), cd_t(0.15599519, -0.18372943), cd_t(-0.01863849, -0.0137227), cd_t(-0.00862298, -0.05205002)},
        {cd_t(-0.00862298, -0.05205002), cd_t(-0.01863849, -0.0137227), cd_t(0.15599519, -0.18372943), cd_t(-0.00862298, -0.05205002)},
        {cd_t(-0.01863849, -0.0137227), cd_t(-0.00862298, -0.05205002), cd_t(-0.00862298, -0.05205002), cd_t(0.15599519, -0.18372943)}};

    double eps;
    for (size_t i = 0; i < goodResult0.n_rows; i++)
        for (size_t j = 0; j < goodResult0.n_cols; j++)
        {
            //std::cout << "i,j = " << i << " " << j << " ";
            eps = 0.0;
            eps += std::abs(1.0 - goodResult0(i, j).real() / greenCluster0Mat.data()(i, j, 0).real());
            eps += std::abs(1.0 - goodResult1(i, j).real() / greenCluster0Mat.data()(i, j, 1).real());
            eps += std::abs(1.0 - goodResult0(i, j).imag() / greenCluster0Mat.data()(i, j, 0).imag());
            eps += std::abs(1.0 - goodResult1(i, j).imag() / greenCluster0Mat.data()(i, j, 1).imag());
            //std::cout << "eps = " << eps << " ";
            ASSERT_TRUE(DELTA > eps);
            ASSERT_NEAR(greenCluster0Mat.data()(i, j, 0).real(), goodResult0(i, j).real(), DELTA);
            ASSERT_NEAR(greenCluster0Mat.data()(i, j, 1).imag(), goodResult1(i, j).imag(), DELTA);
            //std::cout << "eps = " << eps << std::endl;
        }

    //Test moments:
    for (size_t i = 0; i < greenCluster0Mat.data().n_cols; i++)
    {
        for (size_t j = 0; j < greenCluster0Mat.data().n_rows; j++)
        {

            std::cout << "greenCluster0Mat.sm()(i, j).real() = " << greenCluster0Mat.sm()(i, j).real() << std::endl;
            ASSERT_DOUBLE_EQ(greenCluster0Mat.zm()(i, j).real(), 0.0);
            if (i == j)
            {
                ASSERT_DOUBLE_EQ(greenCluster0Mat.fm()(i, j).real(), 1.0);
                //ASSERT_DOBULE_EQ(greenCluster0Mat.sm()(i, j).real(), )
            }
        }
    }
}

TEST(GreenMatTest, Moments)
{

    auto greenCluster0Mat = BuildGreenMat();

    ClusterMatrixCD_t sm(4, 4);
    ClusterMatrixCD_t tm(4, 4);
    sm.zeros();
    tm.zeros();

    sm(0, 1) = sm(1, 0) = sm(1, 3) = sm(3, 1) = sm(2, 3) = sm(3, 2) = sm(0, 2) = sm(2, 0) = 0.90;
    sm(0, 0) = sm(1, 1) = sm(2, 2) = sm(3, 3) = -mu;

    tm(0, 1) = tm(1, 0) = tm(1, 3) = tm(3, 1) = tm(2, 3) = tm(3, 2) = tm(0, 2) = tm(2, 0) = -5.292;
    tm(0, 0) = tm(1, 1) = tm(2, 2) = tm(3, 3) = 12.0636;
    tm(0, 3) = tm(1, 2) = tm(2, 1) = tm(3, 0) = 1.62;

    for (size_t i = 0; i < greenCluster0Mat.data().n_cols; i++)
    {
        for (size_t j = 0; j < greenCluster0Mat.data().n_rows; j++)
        {

            std::cout << "greenCluster0Mat.tm()(i, j).real() = " << greenCluster0Mat.tm()(i, j).real() << std::endl;
            ASSERT_DOUBLE_EQ(greenCluster0Mat.zm()(i, j).real(), 0.0);
            ASSERT_DOUBLE_EQ(greenCluster0Mat.zm()(i, j).imag(), 0.0);
            ASSERT_DOUBLE_EQ(greenCluster0Mat.fm()(i, j).imag(), 0.0);
            if (i == j)
            {
                ASSERT_DOUBLE_EQ(greenCluster0Mat.fm()(i, j).real(), 1.0);
                ASSERT_DOUBLE_EQ(greenCluster0Mat.fm()(i, j).imag(), 0.0);
            }
            else
            {
                ASSERT_DOUBLE_EQ(greenCluster0Mat.fm()(i, j).real(), 0.0);
                ASSERT_DOUBLE_EQ(greenCluster0Mat.fm()(i, j).imag(), 0.0);
            }
            ASSERT_NEAR(greenCluster0Mat.sm()(i, j).real(), sm(i, j).real(), DELTA);
            ASSERT_NEAR(greenCluster0Mat.sm()(i, j).imag(), 0.0, DELTA);

            ASSERT_NEAR(greenCluster0Mat.tm()(i, j).real(), tm(i, j).real(), DELTA);
            ASSERT_NEAR(greenCluster0Mat.tm()(i, j).imag(), 0.0, DELTA);
        }
    }
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

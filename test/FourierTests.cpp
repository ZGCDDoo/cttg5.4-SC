

#include <gtest/gtest.h>
#include "../src/Includes/Utilities/Fourier.hpp"
#include "../src/Includes/Utilities/GreenMat.hpp"
//#include "../src/Includes/ModelTriangle2x2.hpp"

//TO DO MOve greenMat and GreenTau tests to seperate files.

const double DELTA = 1e-7;

TEST(FourierTest, Init)
{

    //Test for non-interacting Green fct with tau > 0 and no dispersion relation, just a chemical potential
    const double mu = 0.0;
    const double beta = 10.1;
    const size_t NMat = 10000000;

    SiteVectorCD_t greenMat(NMat);
    for (size_t i = 0; i < NMat; i++)
    {
        double wn = (2.0 * i + 1.0) * M_PI / beta;
        cd_t tmp = cd_t(mu, wn);
        greenMat(i) = (1.0 / tmp);
    }
    //std::cout << "greenMat = " << greenMat << std::endl;
    double tau = beta / 2.0;
    double greenTau = Fourier::MatToTau(greenMat, tau, beta);
    double greenTauGood = -std::exp(mu * tau) * (1.0 - 1.0 / (1.0 + std::exp(-beta * mu)));

    std::cout << "greenTau = " << greenTau << std::endl;
    std::cout << "greenTauGood = " << greenTauGood << std::endl;
    ASSERT_NEAR(greenTauGood, greenTau, DELTA);
}

TEST(FourierTest, MatToTauCluster)
{
    //simple diagonal green function to test the fourier transform with the moments.
    //G is 2x2 diagonal G_00 = (iwn + mu) => Tloc = hyb = 0

    const size_t NMat = 30;
    ClusterMatrixCD_t fmhyb(2, 2);
    ClusterMatrixCD_t tLoc(2, 2);
    ClusterCubeCD_t hybdata(2, 2, NMat);
    fmhyb.zeros();
    tLoc.zeros();
    hybdata.zeros();

    const double mu = 0.10; //2.94;
    const double beta = 10.1;

    // for (size_t n = 0; n < hybdata.n_slices; n++)
    // {
    //     cd_t iwn = cd_t(0.0, (2.0 * n + 1) * M_PI / beta);
    // }

    GreenMat::HybridizationMat hybMat(hybdata, fmhyb);
    GreenMat::GreenCluster0Mat greenCluster0Mat(hybMat, tLoc, mu, beta);
    //std::cout << "hybMat.data().print(); = " << std::endl;
    //hybMat.data().print();
    //std::cout << "greenCluster0Mat.data().print(); = " << std::endl;
    //greenCluster0Mat.data().print();

    //std::cout << "greenCluster0Mat moments: " << std::endl;
    //greenCluster0Mat.zm().print();
    //greenCluster0Mat.fm().print();
    //greenCluster0Mat.sm().print();
    //greenCluster0Mat.tm().print();

    const double tau = beta / 20.330;
    ClusterMatrix_t greenCluster0Tau = Fourier::MatToTauCluster(greenCluster0Mat, tau);
    //std::cout << "greenCluster0Tau.print(); = " << std::endl;

    double goodResult = -0.28057847825125032;
    //std::cout << "g(0,0) - goodResult " << greenCluster0Tau(0, 0) - goodResult << std::endl;

    double eps = std::abs(1.0 - goodResult / greenCluster0Tau(0, 0));
    eps += std::abs(1.0 - greenCluster0Tau(0, 0) / greenCluster0Tau(1, 1));
    ASSERT_TRUE(DELTA > eps);
    ASSERT_NEAR(goodResult, greenCluster0Tau(0, 0), DELTA);
}

// TEST(FourierTest, )
// {
//     std::ifstream fin("testtriangle.json");
//     Json jj;
//     fin >> jj;
//     Models::ModelTriangle_2D<2, 2> modelTriangle(jj);
//     GreenMat::GreenCluster0Mat greenCluster0Mat = modelTriangle.greenCluster0Mat();
//     const size_t NTau = 10000;
//     double beta = modelTriangle.beta();
//     cd_t iwn(0.0, M_PI / beta); //first frequency

//     GreenTau::GreenCluster0Tau greenCluster0Tau(greenCluster0Mat, NTau);
//     cd_t gfiwn = Fourier::TauToMat(greenCluster0Tau.data().tube(0, 0), iwn, beta);
//     std::cout << "matTotau and back : gfiwn = " << gfiwn << std::endl;
//     std::cout << " greenCluster0Mat.data()(0,0,0)  = " << greenCluster0Mat.data()(0, 0, 0) << std::endl;
// }

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

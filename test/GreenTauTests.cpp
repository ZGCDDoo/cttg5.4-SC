#include <gtest/gtest.h>
#include "../src/Includes/Utilities/GreenMat.hpp"
#include "../src/Includes/Utilities/GreenTau.hpp"
#include "../src/Includes/Utilities/Fourier.hpp"

const double DELTADIAG = 1e-2;
const double DELTANONDIAG = 1e-2;
const double DELTANONINTERAC = 2e-5;
const double BETA = 10.1;
const double EPS = 1e-12;
const double MU = 2.94;
const size_t NTAU = 100000;
const size_t Nc = 4;
const double t = 0.9;
const double xi1 = -MU + 2.0 * t;
const double xi2 = -(MU + 2.0 * t);
const double energy = MU + 0.1;

//const IO::IOSquare2x2 IOMODEL;
using GreenTau_t = GreenTau::GreenCluster0Tau<IO::IOSquare2x2>;

GreenMat::GreenCluster0Mat BuildGreenMat()
{
    ClusterMatrixCD_t tLoc(4, 4);
    ClusterMatrixCD_t fmhyb(4, 4);
    tLoc.zeros();
    fmhyb.zeros();

    tLoc(0, 1) = tLoc(1, 0) = tLoc(1, 3) = tLoc(3, 1) = tLoc(2, 3) = tLoc(3, 2) = tLoc(0, 2) = tLoc(2, 0) = t;
    ClusterCubeCD_t hybdata(4, 4, 30);
    hybdata.zeros();

    GreenMat::HybridizationMat hybMat(hybdata, fmhyb); //HYbridation nulle

    GreenMat::GreenCluster0Mat greenCluster0Mat(hybMat, tLoc, MU, BETA);
    return greenCluster0Mat;
}

GreenMat::GreenCluster0Mat BuildGreenMatNonInteracting()
{
    ClusterMatrixCD_t tLoc(4, 4);
    ClusterMatrixCD_t fmhyb(4, 4);
    tLoc.zeros();
    fmhyb.zeros();
    tLoc(0, 0) = tLoc(1, 1) = tLoc(2, 2) = tLoc(3, 3) = energy;

    ClusterCubeCD_t hybdata(4, 4, 30);
    hybdata.zeros();

    GreenMat::HybridizationMat hybMat(hybdata, fmhyb); //HYbridation nulle

    GreenMat::GreenCluster0Mat greenCluster0Mat(hybMat, tLoc, MU, BETA);
    return greenCluster0Mat;
}

double Fermi(const double &beta, const double &xi)
{
    return (1.0 / (std::exp(beta * xi) + 1.0));
}

double greenTau0(const double &mu, const double &energy, const double &tau, const double &beta)
{
    double green;
    double xi = energy - mu;
    if (tau > 0)
    {
        green = -std::exp(-xi * tau) * (1.0 - Fermi(beta, xi));
    }
    else
    {
        green = std::exp(-xi * tau) * Fermi(beta, xi);
    }
    return green;
}

TEST(GreenTauTests, Init)
{

    //simple diagonal green function to test the fourier transform with the moments.
    //G is 2x2 diagonal G_00 = (iwn + mu) => Tloc = hyb = 0

    GreenMat::GreenCluster0Mat greenCluster0Mat = BuildGreenMat();
    GreenTau_t greenCluster0Tau(greenCluster0Mat, NTAU);

    //Test diagonal part
    std::cout << "======Start Init======== " << std::endl;
    std::cout << "======     Start Diagonal part======== " << std::endl;
    std::cout << " greenCluster0Tau(0, 0, NTAU - 1) =  " << greenCluster0Tau(0, 0, BETA - EPS) << std::endl;
    std::cout << " goodvalue at tau = beta = " << -1.0 << std::endl;
    std::cout << "\n";
    std::cout << " greenCluster0Tau(0, 0, NTAU/2) =  " << greenCluster0Tau(0, 0, BETA / 2.0) << std::endl;
    std::cout << " goodvalue at tau = beta/2 = " << -0.0007903151368341482 << std::endl;
    std::cout << "=======    End Diagonal Part======== " << std::endl;
    std::cout << "\n";

    double eps;
    for (size_t i = 0; i < Nc; i++)
    {
        eps = std::abs(1.0 - greenCluster0Tau(0, 0, BETA - EPS) / (-1.0));
        ASSERT_TRUE(eps < DELTADIAG);
        eps = std::abs(1.0 - greenCluster0Tau(0, 0, BETA / 2.0) / (-0.0007903151368341482));
        ASSERT_TRUE(eps < DELTADIAG);
    }

    //Test non-diagonal part
    const double tau1 = BETA / 2.0;
    double goodGreenTau1 = -(std::exp(xi1 * (BETA - tau1)) * Fermi(BETA, xi1) - std::exp(xi2 * (BETA - tau1)) * Fermi(BETA, xi2)) / 4.0; //out-of-diagonal, but not anti-diagonal

    std::cout << "======     Start Non-Diagonal part======== " << std::endl;
    std::cout << " greenCluster0Tau(0, 1, beta/2) =  " << greenCluster0Tau(0, 1, BETA / 2.0) << std::endl;
    std::cout << " goodvalue at tau = beta/2 = " << goodGreenTau1 << std::endl;
    std::cout << "=======    End Non-Diagonal Part======== " << std::endl;
    std::cout << "======End Init======== " << std::endl;
    std::cout << "\n";

    eps = std::abs(1.0 - greenCluster0Tau(0, 1, BETA / 2.0) / (goodGreenTau1));
    ASSERT_TRUE(eps < DELTANONDIAG);
    eps = std::abs(1.0 - greenCluster0Tau(0, 1, BETA / 2.0) / (goodGreenTau1));
    ASSERT_TRUE(eps < DELTANONDIAG);
    eps = std::abs(1.0 - greenCluster0Tau(0, 1, BETA / 2.0) / (goodGreenTau1));
    ASSERT_TRUE(eps < DELTANONDIAG);
    eps = std::abs(1.0 - greenCluster0Tau(0, 1, BETA / 2.0) / (goodGreenTau1));
    ASSERT_TRUE(eps < DELTANONDIAG);
}

TEST(GreenTauTests, NonInteracting)
{
    GreenMat::GreenCluster0Mat greenCluster0Mat = BuildGreenMatNonInteracting();
    GreenTau_t greenCluster0Tau(greenCluster0Mat, NTAU);

    double tau = -1e-10;
    double tau1 = 1e-9;
    double tau2 = BETA / 2.0;
    double tau3 = -BETA / 3.0;
    double tau4 = BETA - tau1;
    double eps;

    std::cout << "======Start non-interacting======== " << std::endl;
    std::cout << " greenCluster0Tau(0, 0, -1e-10) =  " << greenCluster0Tau(0, 0, tau) << std::endl;
    std::cout << " goodvalue at tau 1e-10 = " << greenTau0(MU, energy, tau, BETA) << std::endl;
    std::cout << "\n";
    std::cout << " greenCluster0Tau(0, 0, 1e-9) =  " << greenCluster0Tau(0, 0, tau1) << std::endl;
    std::cout << " goodvalue at tau1 = " << greenTau0(MU, energy, tau1, BETA) << std::endl;
    std::cout << "\n";
    std::cout << " greenCluster0Tau(0, 0, beta/2.0) =  " << greenCluster0Tau(0, 0, tau2) << std::endl;
    std::cout << " goodvalue at beta/2.0 = " << greenTau0(MU, energy, tau2, BETA) << std::endl;
    std::cout << "\n";
    std::cout << " greenCluster0Tau(0, 0, -beta/3.0) =  " << greenCluster0Tau(0, 0, tau3) << std::endl;
    std::cout << " goodvalue at -beta/3.0 = " << greenTau0(MU, energy, tau3, BETA) << std::endl;
    std::cout << "\n";
    std::cout << " greenCluster0Tau(0, 0, beta) =  " << greenCluster0Tau(0, 0, tau4) << std::endl;
    std::cout << " goodvalue at beta = " << greenTau0(MU, energy, tau4, BETA) << std::endl;
    std::cout << "\n";
    std::cout << "==========End non-interacting======== " << std::endl;
    std::cout << "\n";

    const double ERR = 1e-7;
    ASSERT_NEAR(greenCluster0Tau(0, 0, tau), greenTau0(MU, energy, tau, BETA), ERR);
    ASSERT_NEAR(greenCluster0Tau(0, 0, tau1), greenTau0(MU, energy, tau1, BETA), ERR);
    ASSERT_NEAR(greenCluster0Tau(0, 0, tau2), greenTau0(MU, energy, tau2, BETA), ERR);
    ASSERT_NEAR(greenCluster0Tau(0, 0, tau3), greenTau0(MU, energy, tau3, BETA), ERR);
    ASSERT_NEAR(greenCluster0Tau(0, 0, tau4), greenTau0(MU, energy, tau4, BETA), ERR);

    eps = std::abs(1.0 - greenCluster0Tau(0, 0, tau) / greenTau0(MU, energy, tau, BETA));
    ASSERT_TRUE(eps < DELTANONINTERAC);
    eps = std::abs(1.0 - greenCluster0Tau(0, 0, tau1) / greenTau0(MU, energy, tau1, BETA));
    ASSERT_TRUE(eps < DELTANONINTERAC);
    eps = std::abs(1.0 - greenCluster0Tau(0, 0, tau2) / greenTau0(MU, energy, tau2, BETA));
    ASSERT_TRUE(eps < DELTANONINTERAC);
    eps = std::abs(1.0 - greenCluster0Tau(0, 0, tau3) / greenTau0(MU, energy, tau3, BETA));
    ASSERT_TRUE(eps < DELTANONINTERAC);

    std::cout << "first moment = " << -(greenCluster0Tau(0, 0, 1e-12) + greenCluster0Tau(0, 0, BETA - 1e-12)) << std::endl;
}

TEST(GreenTauTests, Operator)
{
    GreenMat::GreenCluster0Mat greenCluster0Mat = BuildGreenMatNonInteracting();
    GreenTau_t greenCluster0Tau(greenCluster0Mat, NTAU);

    double tau = BETA / 1.1167;
    double tau1 = -tau + 0.12345680;
    double goodValue = greenTau0(MU, energy, tau, BETA);
    double goodValue1 = greenTau0(MU, energy, tau1, BETA);
    double eps;

    std::cout << "======Start operator() ======== " << std::endl;
    std::cout << " greenCluster0Tau(0, 0, beta/1.1167) =  " << greenCluster0Tau(0, 0, tau) << std::endl;
    std::cout << " goodvalue at tau = beta/1.167  = " << greenTau0(MU, energy, tau, BETA) << std::endl;
    std::cout << "\n";
    std::cout << " greenCluster0Tau(0, 0, tau1) =  " << greenCluster0Tau(0, 0, tau1) << std::endl;
    std::cout << " goodvalue at tau1  = " << greenTau0(MU, energy, tau1, BETA) << std::endl;
    std::cout << "\n";
    std::cout << "==========End operator() ======== " << std::endl;
    std::cout << "\n";

    for (size_t i = 0; i < Nc; i++)
    {
        eps = std::abs(1.0 - greenCluster0Tau(i, i, tau) / goodValue);
        ASSERT_TRUE(eps < DELTANONINTERAC);
        eps = std::abs(1.0 - greenCluster0Tau(i, i, tau1) / goodValue1);
        ASSERT_TRUE(eps < DELTANONINTERAC);
    }
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
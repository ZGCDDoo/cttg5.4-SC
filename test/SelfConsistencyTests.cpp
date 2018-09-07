#include <gtest/gtest.h>
#include "../src/Includes/Utilities/SelfConsistency.hpp"
#include "../src/Includes/Models/H0Square.hpp"
#include "../src/Includes/Models/SIAM_Square.hpp"

const double DELTA = 1e-5;
const double BETA = 10.1;
const double MU = 1.5;
const size_t NMAT = 20;
const double hybFM = 4.0;
const double hartree = 0.1;
const double fock = 0.312;
const std::string FNAME_HYB = "../test/data/DMFT/hybfm_SIAM_Square.dat";
const std::string FNAME_JSON = "../test/data/DMFT/test_dmft0.json";

ClusterCubeCD_t BuildGreenImpurity()
{

    //Green impurity given by gimp = inverse(iwn + mu - hyb - self)
    //hyb = fm/iwn ; self given by
    ClusterCubeCD_t greenImpurity(1, 1, NMAT);
    ClusterCubeCD_t hybMat(1, 1, NMAT);
    hybMat.zeros();
    cd_t iwn;
    for (size_t n = 0; n < NMAT; n++)
    {
        iwn = cd_t(0.0, (2.0 * n + 1.0) * M_PI / BETA);
        hybMat(0, 0, n) = hybFM / iwn;
        greenImpurity(0, 0, n) = 1.0 / (iwn + MU - hybMat(0, 0, n) - (hartree + fock / iwn));
    }

    //hybMat.save(FNAME_HYB, arma::raw_ascii);
    return greenImpurity;
}

TEST(SelfConsistencyTests, SelfConsistency)
{

    //     //Green impurity given by gimp = inverse(iwn + mu - hyb - self)
    //     //hyb = 0.0 ; self = hartree + fock/iwn; Tested numerically with Mathematica.
    std::ifstream fin(FNAME_JSON);
    Json jj;
    fin >> jj;
    fin.close();
    std::cout << "before greenImpurity" << std::endl;
    ClusterCubeCD_t greenImpurity = BuildGreenImpurity();
    Models::SIAM_Square siamsquare(jj);
    //Build The dummy nmatrixSigma
    ClusterMatrix_t nSigma(1, 1);
    nSigma.zeros();
    nSigma.save("nUpMatrix.dat");
    nSigma.save("nDownMatrix.dat");
    SelfCon::SelfConsistency<IO::IOSIAM, Models::SIAM_Square, Models::H0Square<1, 1>> selfcon(jj, siamsquare, greenImpurity, FermionSpin_t::Up);
    selfcon.DoSCGrid();

    cd_t hybNext0 = cd_t(0.309056, -1.25789);
    cd_t hybNext10 = cd_t(0.0937823, -0.534391);
    std::cout << "selfcon.hybNext(0, 0, 0).imag() = " << selfcon.hybNext()(0, 0, 0).imag() << std::endl;
    std::cout << "selfcon.hybNext(0, 0, 10).imag() = " << selfcon.hybNext()(0, 0, 10).imag() << std::endl;
    ASSERT_NEAR(hybNext0.real(), selfcon.hybNext()(0, 0, 0).real(), DELTA);
    ASSERT_NEAR(hybNext0.imag(), selfcon.hybNext()(0, 0, 0).imag(), DELTA);
    ASSERT_NEAR(hybNext10.real(), selfcon.hybNext()(0, 0, 10).real(), DELTA);
    ASSERT_NEAR(hybNext10.imag(), selfcon.hybNext()(0, 0, 10).imag(), DELTA);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

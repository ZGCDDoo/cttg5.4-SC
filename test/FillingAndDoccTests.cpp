#define SUBMATRIX

#include <gtest/gtest.h>

#include "../src/Includes/IS/Obs/FillingAndDocc.hpp"
#include "../src/Includes/Models/ModelSquare2x2.hpp"

using Model_t = Models::ModelSquare2x2;
using IOModel_t = IO::IOSquare2x2;
using FillingAndDocc_t = Markov::Obs::FillingAndDocc<IO::IOSquare2x2, Models::ModelSquare2x2>;
using ISDataCT_t = Markov::Obs::ISDataCT<IO::IOSquare2x2, Models::ModelSquare2x2>;

// const double DELTA = 1e-11;
const std::string FNAME = "../test/data/cdmft_square2x2/params1.json";

FillingAndDocc_t BuildFillingAndDocc() //for Square2x2
{

    std::ifstream fin(FNAME);
    Json jj;
    fin >> jj;
    fin.close();
    Model_t model(jj);
    std::shared_ptr<ISDataCT_t> dataCT(
        new ISDataCT_t(
            jj["beta"].get<double>(),
            model, jj["NTAU"].get<double>()));

    Utilities::EngineTypeFibonacci3217_t rng(0);
    std::shared_ptr<Utilities::UniformRngFibonacci3217_t> urngPtr(new Utilities::UniformRngFibonacci3217_t(rng, Utilities::UniformDistribution_t(0.0, 1.0)));

    const size_t N_T_INV = 5;
    FillingAndDocc_t fillingAndDocc(dataCT, urngPtr, N_T_INV);
    return fillingAndDocc;
}

TEST(FillingAndDoccTests, Init)
{
    FillingAndDocc_t fillingAndDocc = BuildFillingAndDocc();
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}



#include <gtest/gtest.h>
#include "../src/Includes/Utilities/Utilities.hpp"
#include "../src/Includes/Utilities/LinAlg.hpp"

using namespace Utilities;
using namespace LinAlg;

const double DELTA = 1e-11;
TEST(UtilitiesTest, DotVectors)
{
    //c for complex
    SiteVector_t v1 = {1.0, 11.1, 90.2, -1.999};
    SiteVectorCD_t v1c(v1, 0.88 * v1);
    SiteVector_t v2 = {-1.9760, 7.6, 0.0002, 1.299};
    SiteVectorCD_t v2c(v2, 0.1 * v2);

    ASSERT_NEAR(DotVectors(v1, v2), arma::dot(v1, v2), 1e-10);
    ASSERT_NEAR(DotVectors(v1c, v2c).real(), arma::dot(v1c, v2c).real(), 1e-10);
    ASSERT_NEAR(DotVectors(v1c, v2c).imag(), arma::dot(v1c, v2c).imag(), 1e-10);
}

TEST(UtilitiesTest, MatrixVectorMult)
{
    const size_t NN = 3;
    SiteVector_t resultBlas(NN);
    SiteVectorCD_t resultBlasc(NN);
    SiteVector_t v1 = {1.0, 11.1, 90.2};
    SiteVectorCD_t v1c(v1, 0.88 * v1);
    ClusterMatrix_t A = {{-1.9760, 7.6, 0.0002}, {0.12, 1.299, -5.3}, {0.005, 0.02, 0.03}};
    ClusterMatrixCD_t Ac(-0.223 * A, A);
    double alpha = -0.217;
    SiteVector_t good = alpha * A * v1;
    cd_t alphac(1.1, 0.09);
    SiteVectorCD_t goodc = alphac * Ac * v1c;

    Matrix_t AMatrix(A);
    MatrixVectorMult(AMatrix, v1, alpha, resultBlas);
    MatrixCD_t AcMatrix(Ac);
    MatrixVectorMult(AcMatrix, v1c, alphac, resultBlasc);
    for (size_t ii = 0; ii < NN; ii++)
    {
        ASSERT_NEAR(good(ii), resultBlas(ii), 1e-10);
        ASSERT_NEAR(goodc(ii).real(), resultBlasc(ii).real(), 1e-10);
        ASSERT_NEAR(goodc(ii).imag(), resultBlasc(ii).imag(), 1e-10);
    }
}

TEST(UtilitiesTest, Dot)
{
    const size_t NN = 3;
    SiteVector_t resultBlas(NN);
    SiteVector_t v1 = {1.0, 11.1, 90.2};
    SiteVector_t v2 = {-1.1, 0.00012, -0.16667};
    ClusterMatrix_t A = {{-1.9760, 7.6, 0.0002}, {0.12, 1.299, -5.3}, {0.005, 0.02, 0.03}};
    double dotGood = arma::dot(v2, A * v1);

    Matrix_t AMatrix(A);
    AMatrix.Resize(100, 100);
    AMatrix.Resize(NN, NN);
    double dotTest = Dot(v2, AMatrix, v1);
    ASSERT_DOUBLE_EQ(dotTest, dotGood);
}

TEST(UtilitiesTest, VectorMatrixMult)
{
    std::cout << "start vectorMatrixMUlt Test" << std::endl;
    const size_t NN = 3;
    SiteVector_t resultBlas(NN);
    SiteVectorCD_t resultBlasc(NN);
    SiteVector_t v1 = {1.0, 11.1, 90.2};
    SiteVectorCD_t v1c(v1, 2.1 * v1);
    //SiteRowCD_t v1c(v1, v1;
    ClusterMatrix_t A = {{-1.9760, 7.6, 0.0002}, {0.12, 1.299, -5.3}, {0.005, 0.02, 0.03}};
    ClusterMatrixCD_t Ac(A, 0.7 * A);
    double alpha = 1.1;
    cd_t alphac(0.1, 3.03);
    std::cout << "In vectorMatrixMUlt Test before transpose" << std::endl;

    SiteRow_t r1(NN);
    std::cout << "In vectorMatrixMUlt Test after r1" << std::endl;
    for (size_t i = 0; i < NN; i++)
    {
        r1(i) = v1(i);
    }
    SiteRow_t good = alpha * r1 * A;

    std::cout << "In vectorMatrixMUlt Test" << std::endl;
    SiteRowCD_t r1cR(NN);
    r1cR(0) = v1c(0);
    r1cR(1) = v1c(1);
    r1cR(2) = v1c(2);
    SiteRowCD_t goodc = alphac * r1cR * Ac;
    std::cout << "In vectorMatrixMUlt Test after goodc" << std::endl;

    Matrix_t AMatrix(A);
    MatrixCD_t AcMatrix(Ac);
    AMatrix.Resize(100, 100);
    AMatrix.Resize(NN, NN);
    AcMatrix.Resize(301, 301);
    AcMatrix.Resize(NN, NN);

    VectorMatrixMult(v1, AMatrix, alpha, resultBlas);
    VectorMatrixMult(v1c, AcMatrix, alphac, resultBlasc);
    // resultBlasc.print();
    // std::cout << "\n"
    //           << std::endl;
    // goodc.print();

    for (size_t ii = 0; ii < NN; ii++)
    {
        ASSERT_NEAR(good(ii), resultBlas(ii), 1e-10);
        ASSERT_NEAR(goodc(ii).real(), resultBlasc(ii).real(), 1e-10);
        ASSERT_NEAR(goodc(ii).imag(), resultBlasc(ii).imag(), 1e-10);
    }
}

TEST(UtilitiesTest, DGEMM)
{

    ClusterMatrix_t A(5, 10);
    ClusterMatrix_t B(10, 11);
    ClusterMatrix_t CGood(5, 11);
    ClusterMatrix_t CTest(5, 11);
    const double alpha = -0.1247;
    const double beta = 1.19;

    A.randu();
    B.randn();
    CTest.randn();
    CTest += 1.1;
    CGood = CTest;

    CGood = alpha * A * B + beta * CGood;

    Matrix_t AMatrix(A);
    Matrix_t BMatrix(B);
    Matrix_t CTestMatrix(CTest);

    AMatrix.Resize(100, 100);
    AMatrix.Resize(5, 10);
    BMatrix.Resize(301, 301);
    BMatrix.Resize(10, 11);
    CTestMatrix.Resize(100, 100);
    CTestMatrix.Resize(5, 11);

    DGEMM(alpha, beta, AMatrix, BMatrix, CTestMatrix);

    for (size_t ii = 0; ii < CGood.n_rows; ii++)
    {
        for (size_t jj = 0; jj < CGood.n_cols; jj++)
        {
            ASSERT_NEAR(CGood(ii, jj), CTestMatrix(ii, jj), 1e-10);
        }
    }
}

TEST(UtilitiesTest, Vertex)
{

    Utilities::Vertex vertex(1.1, 2, AuxSpin_t::Up);
    ASSERT_DOUBLE_EQ(vertex.tau(), 1.1);
    ASSERT_EQ(vertex.site(), 2);
    ASSERT_EQ(vertex.aux(), AuxSpin_t::Up);

    vertex.FlipAux();
    ASSERT_EQ(vertex.aux(), AuxSpin_t::Down);
}

TEST(UtilitiesTest, TriangularSolve)
{

    //=============Test Solve =====================
    ClusterMatrix_t A(10, 10);
    A.randn();
    ClusterMatrix_t Aupper = A;
    ClusterMatrix_t Alower = Aupper;
    for (size_t i = 0; i < Alower.n_rows; i++)
    {
        Alower(i, i) = 1.0;
    }

    SiteVector_t B(10);
    B.randn();

    SiteVector_t goodupper = arma::solve(arma::trimatu(Aupper), B);
    SiteVector_t goodlower = arma::solve(arma::trimatl(Alower), B);
    SiteVector_t testupper = B;
    SiteVector_t testlower = B;

    Matrix_t AMatrix(A);
    TriangularSolve('u', 'n', AMatrix, testupper);
    TriangularSolve('l', 'n', AMatrix, testlower);

    for (size_t i = 0; i < B.n_rows; i++)
    {
        ASSERT_NEAR(goodupper(i), testupper(i), 1e-10);
        ASSERT_NEAR(goodlower(i), testlower(i), 1e-10);
    }

    //=============Test inverse =====================
    Matrix_t AupperInverse(arma::trimatu(Aupper).i()); //The good inverse matrixs
    Matrix_t AlowerInverse(arma::trimatl(Alower).i());
    Matrix_t AM(A);
    TriangularInverse('u', AM);
    TriangularInverse('l', AM);

    Matrix_t Atestu(arma::trimatu(AM.mat()));
    Matrix_t Atestl(arma::trimatl(AM.mat()));

    for (size_t i = 0; i < AupperInverse.n_rows(); i++)
    {
        ASSERT_NEAR(Atestu(i, i), Atestl(i, i), 1e-10);
        Atestl(i, i) = 1.0;
        for (size_t j = 0; j < AupperInverse.n_cols(); j++)
        {
            ASSERT_NEAR(Atestu(i, j), AupperInverse(i, j), 1e-10);
            ASSERT_NEAR(Atestl(i, j), AlowerInverse(i, j), 1e-10);
        }
    }
}

TEST(UtilitiesTest, LUExtractLUInverse)
{
    const size_t NN = 3;
    ClusterMatrix_t A = {{1.0, 2.0, 4.0}, {3, 2, 2}, {2, 1, 3}};
    ClusterMatrix_t Alower = arma::trimatl(A);
    ClusterMatrix_t Aupper = arma::trimatu(A);
    //set Alower diagonal to one
    for (size_t ii = 0; ii < NN; ii++)
    {

        Alower(ii, ii) = 1.0;
    }

    Matrix_t AM(A);
    Matrix_t AMlower(NN, NN);
    Matrix_t AMupper(NN, NN);

    ExtractLU(AM, AMlower, AMupper);

    for (size_t i = 0; i < AM.n_rows(); i++)
    {
        for (size_t j = 0; j < AM.n_cols(); j++)
        {
            ASSERT_NEAR(Alower(i, j), AMlower(i, j), 1e-10);
            ASSERT_NEAR(Aupper(i, j), AMupper(i, j), 1e-10);
        }
    }

    LUInverse(1.0, AMlower, AMupper, AM.n_rows());
    ClusterMatrix_t goodLUInverse = Aupper.i() * Alower.i();

    for (size_t i = 0; i < AM.n_rows(); i++)
    {
        for (size_t j = 0; j < AM.n_rows(); j++)
        {
            ASSERT_NEAR(AMupper(i, j), goodLUInverse(i, j), DELTA);
        }
    }
}

TEST(UtilitiesTest, BlockRankOneUpgrade)
{
    //Now test the following: I have the matrix a1 2x2, its inverse is m1. I a1 update it so that it is 3x3 by adding a Row r1 and a Column c1.
    //Now I want this new matrix a2 and its inverse m2. I can do this by twice the shermann morrison. Lets test that.

    const size_t k = 3;
    ClusterMatrix_t a1 = {
        {0.19, 1.1, 2.17},
        {-1.272, 3.0, 0.67},
        {0.1, 0.22, 0.367}};

    ClusterMatrix_t m1 = a1.i();

    const SiteVector_t Q = {1.23, -0.221, -0.543}; //new col
    const SiteVector_t R = {1.179, -2.31, 0.321};  //new row

    ClusterMatrix_t a2 = {
        {0.19, 1.1, 2.17, 1.23},
        {-1.272, 3.0, 0.67, -0.221},
        {0.1, 0.22, 0.367, -0.543},
        {1.179, -2.31, 0.321, -1.4}};
    ClusterMatrix_t m2Good = a2.i(); //The good inverse calculated normally.

    const double STildeGood = m2Good(k, k);

    Matrix_t m1Matrix(m1);
    m1Matrix.Resize(100, 100);
    m1Matrix.Resize(k, k);

    SiteVector_t mkQ(k);
    MatrixVectorMult(m1, Q, 1.0, mkQ);
    const double S = -1.4;
    const double STilde = 1.0 / (S - DotVectors(R, mkQ));
    ASSERT_NEAR(STilde, STildeGood, DELTA);
    BlockRankOneUpgrade(m1Matrix, mkQ, R, STilde);

    // //Now m1Matrix shoukld contain the inverse of a2.

    for (size_t i = 0; i < m1Matrix.n_rows(); i++)
    {
        for (size_t j = 0; j < m1Matrix.n_cols(); j++)
        {
            //std::cout << "i ,j = " << i << " " << j << std::endl;
            ASSERT_NEAR(m1Matrix(i, j), m2Good(i, j), DELTA);
        }
    }
}

TEST(UtilitiesTest, BlockRankTwoUpgrade)
{
    //Now test the following: I have the matrix a1 2x2, its inverse is m1. I a1 update it so that it is 3x3 by adding a Row r1 and a Column c1.
    //Now I want this new matrix a2 and its inverse m2. I can do this by twice the shermann morrison. Lets test that.

    const size_t k = 3;
    ClusterMatrix_t a1 = {
        {0.19, 1.1, 2.17},
        {-1.272, 3.0, 0.67},
        {0.1, 0.22, 0.367}};

    ClusterMatrix_t m1 = a1.i();

    Matrix_t Q = {{1.23, -1.17}, {-0.221, 1.17}, {-0.543, 2.99}}; //new cols
    Matrix_t R = {{1.179, -2.31, 0.321}, {9.9, -0.0001, 4.5}};    //new rows
    Matrix_t S = {{6.65, -6.66}, {0.99, 9.1}};

    ClusterMatrix_t a2 = {
        {0.19, 1.1, 2.17, 1.23, -1.17},
        {-1.272, 3.0, 0.67, -0.221, 1.17},
        {0.1, 0.22, 0.367, -0.543, 2.99},
        {1.179, -2.31, 0.321, 6.55, -6.66},
        {9.9, -0.0001, 4.5, 0.99, 9.1}};

    ClusterMatrix_t m2Good = a2.i(); //The good inverse calculated normally.

    Matrix_t m1Matrix(m1);
    m1Matrix.Resize(100, 100);
    m1Matrix.Resize(k, k);

    Matrix_t STilde(2, 2);
    STilde(0, 0) = m2Good(k, k);
    STilde(0, 1) = m2Good(k, k + 1);
    STilde(1, 0) = m2Good(k + 1, k);
    STilde(1, 1) = m2Good(k + 1, k + 1);

    BlockRankTwoUpgrade(m1Matrix, Q, R, STilde);

    // //Now m1Matrix shoukld contain the inverse of a2.

    for (size_t i = 0; i < m1Matrix.n_rows(); i++)
    {
        for (size_t j = 0; j < m1Matrix.n_cols(); j++)
        {
            std::cout << "i ,j = " << i << " " << j << std::endl;
            ASSERT_NEAR(m1Matrix(i, j), m2Good(i, j), DELTA);
        }
    }
}

TEST(UtilitiesTest, ExtractRowAndCol)
{
    const size_t k = 4;
    ClusterMatrix_t a1 = {
        {0.19, 1.1, 1.23, 0.79},
        {-1.272, 3.0, -0.221, 0.51},
        {1.179, -2.31, -1.4, 0.01},
        {-0.15, 0.82, 0.35, -0.43}};

    SiteVector_t v1Col = a1.col(2);
    SiteVector_t v1Row = {-0.15, 0.82, 0.35, -0.43};
    SiteVector_t v1ColTest;
    SiteVector_t v1RowTest;
    //SiteRow_t r1 = a1.row(2);
    Matrix_t a1Matrix(a1);
    a1Matrix.Resize(1000, 1000);
    a1Matrix.Resize(k, k);
    ExtractCol(2, v1ColTest, a1);
    ExtractRow(3, v1RowTest, a1);
    for (size_t i = 0; i < a1Matrix.n_rows(); i++)
    {
        ASSERT_NEAR(v1ColTest(i), v1Col(i), DELTA);
        ASSERT_NEAR(v1RowTest(i), v1Row(i), DELTA);
    }
}

TEST(UtilitiesTest, BlockRankOneDownGrade)
{
    const size_t kk = 40;
    const size_t pp = 11;
    ClusterMatrix_t a1(kk, kk);
    a1.randu();

    ClusterMatrix_t m1 = a1.i();
    const size_t k = m1.n_rows;
    //a2 is obtained from a1 by removing vertex 2 = col and row 2
    ClusterMatrix_t a2 = a1;
    a2.swap_rows(pp, k - 1);
    a2.swap_cols(pp, k - 1);
    a2.shed_row(k - 1);
    a2.shed_col(k - 1);
    ClusterMatrix_t m2Good = a2.i();
    //I need to put the rows and cols to be removed at the end;
    Matrix_t m1Matrix(m1);

    BlockRankOneDowngrade(m1Matrix, pp);

    std::cout << "m2Good " << std::endl;
    m2Good.print();
    std::cout << "m2Test " << std::endl;
    m1Matrix.Print();
    for (size_t i = 0; i < k - 1; i++)
    {
        for (size_t j = 0; j < k - 1; j++)
        {
            // std::cout << "i, j " << i << ", " << j << std::endl;
            ASSERT_NEAR(m2Good(i, j), m1Matrix(i, j), DELTA);
        }
    }
}

TEST(UtilitiesTest, DDMGMM)
{
    const size_t k = 10;
    ClusterMatrix_t A(k, k);
    A.randu();
    Matrix_t AMatrix(A);
    SiteVector_t v1(k);
    v1.randu();
    ClusterMatrix_t Good = arma::diagmat(v1) * A;
    Matrix_t Test;
    DDMGMM(v1, AMatrix, Test);

    for (size_t i = 0; i < Test.n_rows(); i++)
    {
        for (size_t j = 0; j < Test.n_cols(); j++)
        {
            // std::cout << "i, j " << i << ", " << j << std::endl;
            ASSERT_NEAR(Test(i, j), Good(i, j), DELTA);
        }
    }
}

TEST(UtilitiesTest, GetSubMat)
{

    const size_t kk = 6;
    ClusterMatrix_t tmp(kk, kk);
    tmp.randn();
    Matrix_t src(tmp);

    Matrix_t dest = GetSubMat(2, 2, 5, 5, src);
    for (size_t ii = 2; ii < 5; ii++)
    {
        for (size_t jj = 2; jj < 5; jj++)
        {
            ASSERT_DOUBLE_EQ(dest(ii - 2, jj - 2), src(ii, jj));
        }
    }
    dest.Print();

    std::cout << "\n\n\n";
    src.Print();
}

TEST(UtilitiesTest, BlockRankDownGrade)
{
    const size_t kk = 40;
    ClusterMatrix_t a1(kk, kk);
    a1.randu();

    ClusterMatrix_t m1 = a1.i();
    //a2 is obtained from a1 by removing vertex 2 = col and row 2
    // and vertex 11 (counting from 0)
    ClusterMatrix_t a2 = a1;
    a2.swap_rows(kk - 1, 3);
    a2.swap_cols(kk - 1, 3);
    a2.swap_rows(kk - 2, 2);
    a2.swap_cols(kk - 2, 2);
    a2.shed_row(kk - 1);
    a2.shed_col(kk - 1);
    a2.shed_row(kk - 2);
    a2.shed_col(kk - 2);

    ClusterMatrix_t m2Good = a2.i();

    Matrix_t m1Matrix(m1);

    BlockDowngrade(m1Matrix, 2, 2);

    //std::cout << "m2Good " << std::endl;
    //m2Good.print();
    //std::cout << "m2Test " << std::endl;
    //m1Matrix.Print();
    for (size_t i = 0; i < m1Matrix.n_rows(); i++)
    {
        for (size_t j = 0; j < m1Matrix.n_rows(); j++)
        {
            // std::cout << "i, j " << i << ", " << j << std::endl;
            ASSERT_NEAR(m2Good(i, j), m1Matrix(i, j), DELTA);
        }
    }
}

TEST(UtilitiesTest, BlockRankDownGradeVers2)
{
    const size_t kk = 40;
    const size_t pp = 11;
    const size_t nn = 2;
    ClusterMatrix_t a1(kk, kk);
    a1.randu();

    ClusterMatrix_t m1 = a1.i();
    //a2 is obtained from a1 by removing vertex 2 = col and row 2
    // and vertex 11 (counting from 0)
    ClusterMatrix_t a2 = a1;
    a2.swap_rows(kk - 1, pp + 1);
    a2.swap_cols(kk - 1, pp + 1);
    a2.swap_rows(kk - 2, pp);
    a2.swap_cols(kk - 2, pp);
    a2.shed_row(kk - 1);
    a2.shed_col(kk - 1);
    a2.shed_row(kk - 2);
    a2.shed_col(kk - 2);

    ClusterMatrix_t m2Good = a2.i();

    Matrix_t m1Matrix(m1);

    BlockDowngrade(m1Matrix, pp, nn);

    //std::cout << "m2Good " << std::endl;
    //m2Good.print();
    //std::cout << "m2Test " << std::endl;
    //m1Matrix.Print();
    for (size_t i = 0; i < m1Matrix.n_rows(); i++)
    {
        for (size_t j = 0; j < m1Matrix.n_rows(); j++)
        {
            // std::cout << "i, j " << i << ", " << j << std::endl;
            ASSERT_NEAR(m2Good(i, j), m1Matrix(i, j), DELTA);
        }
    }
}

TEST(UtilitiesTest, DotRank2)
{

    const size_t kk = 599;
    ClusterMatrix_t m1Arma(kk, kk);
    m1Arma.randu();
    Matrix_t m1(m1Arma);

    ClusterMatrix_t AArma(kk, kk);
    AArma.randn();
    Matrix_t A(AArma);

    ClusterMatrix_t m2Arma(kk, kk);
    m2Arma.randn();
    Matrix_t m2(m2Arma);

    Matrix_t result = DotRank2(m1, A, m2);
    ClusterMatrix_t good = m1Arma * AArma * m2Arma;

    for (size_t i = 0; i < result.n_rows(); i++)
    {
        for (size_t j = 0; j < result.n_cols(); j++)
        {
            // std::cout << "i, j " << i << ", " << j << std::endl;
            ASSERT_NEAR(result(i, j), good(i, j), DELTA);
        }
    }
}

// TEST(UtilitiesTest, AddOneElementToInverse)
// {
//     ClusterMatrix_t a1 = {
//         {0.19, 1.1, 1.23, 0.79},
//         {-1.272, 3.0, -0.221, 0.51},
//         {1.179, -2.31, -1.4, 0.01},
//         {-0.15, 0.82, 0.35, -0.43}};

//     ClusterMatrix_t m1 = a1.i();
//     const size_t k = m1.n_rows;

//     const double value = -0.123908;
//     a1(2, 2) += value;
//     ClusterMatrix_t m1Good = a1.i();
//     AddOneElementToInverse(m1, 2, value);

//     for (size_t i = 0; i < k - 1; i++)
//     {
//         for (size_t j = 0; j < k - 1; j++)
//         {
//             //std::cout << "In loop " << std::endl;
//             ASSERT_NEAR(m1Good(i, j), m1(i, j), DELTA);
//         }
//     }
// }

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

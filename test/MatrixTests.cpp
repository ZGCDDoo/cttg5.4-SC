

#include <gtest/gtest.h>
#include "../src/Includes/Utilities/Matrix.hpp"

using namespace LinAlg;
using cd_t = std::complex<double>;

// const double DELTA = 1e-11;

TEST(MatrixTest, Init)
{

    Matrix<double> m1(2, 2);
    Matrix<double> m2;
    m2 = m1;
    ASSERT_EQ(m2.n_rows(), 2);
    ASSERT_EQ(m2.n_cols(), 2);

    Matrix<double> m3 = {{0.0, 1.1}, {3.3, -4.4}};

    ASSERT_DOUBLE_EQ(m3(0, 0), 0.0);
    ASSERT_DOUBLE_EQ(m3(0, 1), 1.1);
    ASSERT_DOUBLE_EQ(m3(1, 0), 3.30);
    ASSERT_DOUBLE_EQ(m3(1, 1), -4.4);
}

TEST(MatrixTest, GetElements)
{
    Matrix<double> m1(2, 2);
    m1(0, 0) = 10.0;
    m1(1, 1) = -10.10;
    ASSERT_DOUBLE_EQ(m1(0, 0), 10.0);
    ASSERT_DOUBLE_EQ(m1(1, 1), -10.10);
}

TEST(MatrixTest, ResizeAndSetSize)
{
    Matrix<double> m1(2, 2);
    m1(0, 0) = 10.0;
    m1(1, 1) = -10.10;
    m1.Resize(100, 100);
    m1(10, 10) = -0.001;
    m1(90, 71) = 0.2;
    ASSERT_DOUBLE_EQ(m1(0, 0), 10.0);
    ASSERT_DOUBLE_EQ(m1(1, 1), -10.10);
    ASSERT_DOUBLE_EQ(m1(10, 10), -0.001);
    ASSERT_DOUBLE_EQ(m1(90, 71), 0.2);
}

TEST(MatrixTest, CopyVectorIn)
{
    Matrix<double> m1(1, 1);
    m1.Resize(4, 4);
    m1(4, 4) = 4.4;
    m1(1, 4) = 1.4;
    m1.Resize(10, 10);
    SiteVector_t v1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.10};
    m1.CopyVectorInCol(v1, 2);

    ASSERT_DOUBLE_EQ(m1(4, 4), 4.40);
    ASSERT_DOUBLE_EQ(m1(1, 4), 1.4);
    ASSERT_DOUBLE_EQ(m1(0, 2), 1.0);
    ASSERT_DOUBLE_EQ(m1(1, 2), 2.0);
    ASSERT_DOUBLE_EQ(m1(2, 2), 3.0);
    ASSERT_DOUBLE_EQ(m1(3, 2), 4.0);
    ASSERT_DOUBLE_EQ(m1(4, 2), 5.0);
    ASSERT_DOUBLE_EQ(m1(5, 2), 6.0);

    m1.CopyVectorInRow(v1, 4);
    ASSERT_DOUBLE_EQ(m1(4, 0), 1.0);
    ASSERT_DOUBLE_EQ(m1(4, 7), 8.0);
    ASSERT_DOUBLE_EQ(m1(4, 9), 10.10);

    m1.Print();
}

TEST(MatrixTest, DiagonalMatrix)
{

    SiteVector_t v1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.10};
    Matrix<double> m1 = Matrix<double>::DiagMat(v1);
    arma::Mat<double> m1Good = arma::diagmat(v1);
    Matrix<double> m1Diag = Matrix<double>::DiagMat(10, 1.10);

    for (size_t ii = 0; ii < m1.n_rows(); ii++)
    {
        for (size_t jj = 0; jj < m1.n_cols(); jj++)
        {
            ASSERT_NEAR(m1(ii, jj), m1Good(ii, jj), 1e-10);
        }
    }
    for (size_t i = 0; i < 10; i++)
    {
        ASSERT_NEAR(m1Diag(i, i), 1.1, 1e-10);
    }
}

TEST(MatrixTest, Inverse)
{

    ClusterMatrix_t m1(400, 400);
    m1.randu();
    // m1.print();
    Matrix<double> m1Matrix(m1);
    ClusterMatrix_t m1Inverse = m1.i();
    std::cout << "Inversed Matrix Armadillo" << std::endl;
    m1Matrix.Inverse();
    std::cout << "Inversed Matrix" << std::endl;

    // std::cout << "m1Matrix" << std::endl;
    // m1Matrix.Print();
    // std::cout << "m1Inverse" << std::endl;
    // m1Inverse.print();

    for (size_t ii = 0; ii < m1.n_rows; ii++)
    {
        for (size_t jj = 0; jj < m1.n_cols; jj++)
        {
            ASSERT_NEAR(m1Inverse(ii, jj), m1Matrix(ii, jj), 1e-10);
        }
    }
}

TEST(MatrixTest, AddAndSubstract)
{

    ClusterMatrix_t m1(10, 10);
    m1.randu();
    ClusterMatrix_t m2 = (0.1667234 * m1 - m1 * m1).i();
    ClusterMatrix_t mAdd = m1 + m2;
    ClusterMatrix_t mSubstract = m1 - m2;

    Matrix<double> m1Matrix(m1);
    Matrix<double> m2Matrix(m2);
    Matrix<double> mAddMatrix = m1Matrix + m2Matrix;
    Matrix<double> mSubsctractMatrix = m1Matrix - m2Matrix;

    for (size_t ii = 0; ii < m1.n_rows; ii++)
    {
        for (size_t jj = 0; jj < m1.n_cols; jj++)
        {
            ASSERT_NEAR(mAdd(ii, jj), mAddMatrix(ii, jj), 1e-10);
            ASSERT_NEAR(mSubstract(ii, jj), mSubsctractMatrix(ii, jj), 1e-10);
        }
    }
}

TEST(MatrixTest, MultiplyByScalar)
{

    ClusterMatrix_t m1(10, 10);
    m1.randu();
    const double alpha = -0.1736324;
    Matrix<double> m1Matrix(m1);
    m1 *= alpha;
    m1Matrix *= alpha;

    for (size_t ii = 0; ii < m1.n_rows; ii++)
    {
        for (size_t jj = 0; jj < m1.n_cols; jj++)
        {
            ASSERT_NEAR(m1(ii, jj), m1Matrix(ii, jj), 1e-10);
        }
    }
}

TEST(MatrixTest, SubMat)
{
    Matrix<double> a(4, 4);
    a.Ones();
    Matrix<double> b = a * 1.21;
    b.Resize(2, 2);
    a.SubMat(2, 2, 3, 3, b);
    std::cout << "b= " << std::endl;
    b.Print();
    std::cout << "a= " << std::endl;
    a.Print();
}

TEST(MatrixTest, SubMat_T)
{
    Matrix<double> a(4, 4);
    a.Ones();
    double b = 1.21;
    a.SubMat(0, 0, 1, 1, b);
    std::cout << "a= " << std::endl;
    a.Print();
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

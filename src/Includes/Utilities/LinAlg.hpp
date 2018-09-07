#pragma once

#include "Utilities.hpp"
#include "Matrix.hpp"
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

namespace LinAlg
{

typedef Matrix<double> Matrix_t;
typedef Matrix<cd_t> MatrixCD_t;

void ExtractRow(const size_t &p, SiteVector_t &vec, const Matrix_t &A)
{
    const unsigned int k = A.n_cols();
    const unsigned int ld_A = A.mem_n_rows();
    const unsigned int inc = 1;
    vec.set_size(k);
    dcopy_(&k, &(A.memptr()[p]), &ld_A, vec.memptr(), &inc);
}

void ExtractCol(const size_t &p, SiteVector_t &vec, const Matrix_t &A)
{
    const unsigned int k = A.n_cols();
    const unsigned int ld_A = A.mem_n_rows();
    const unsigned int inc = 1;
    vec.set_size(k);
    dcopy_(&k, &(A.memptr()[p * ld_A]), &inc, vec.memptr(), &inc);
}

// void SubMat(const size_t &r1, const size_t &c1, const size_t &r2, const size_t &c2, const Matrix_t &src, Matrix_t &dest)
// {
//     // std::cerr << "SUbmat problem need to debug" << std::endl;
//     // assert(false);
//     const unsigned int M = r2 - r1;
//     const unsigned int N = c2 - c1;
//     dest.SetSize(M, N);
//     const unsigned int ld_dest = dest.mem_n_rows();
//     const unsigned int ld_src = src.mem_n_rows();
//     const char lo = 'f'; //full matrix

//     dlacpy_(&lo, &M, &N, &(src.memptr()[r1 + c1 * ld_src]), &ld_src, dest.memptr(), &ld_dest);
// }

Matrix_t GetSubMat(const size_t &r1, const size_t &c1, const size_t &r2, const size_t &c2, const Matrix_t &src)
{
    assert(r2 > r1 && c2 > c1);
    const size_t M = r2 - r1;
    const size_t N = c2 - c1;
    Matrix_t submat(M, N);
    for (size_t ii = 0; ii < M; ii++)
    {
        for (size_t jj = 0; jj < N; jj++)
        {
            submat(ii, jj) = src(ii + r1, jj + c1);
        }
    }

    return submat;
}

double DotVectors(const SiteVector_t &v1, const SiteVector_t &v2)
{
    unsigned int N = v1.n_elem;
    assert(v1.n_elem == v2.n_elem);
    unsigned int inc = 1;
    return ddot_(&N, v1.memptr(), &inc, v2.memptr(), &inc);
}

std::complex<double> DotVectors(const SiteVectorCD_t &v1, const SiteVectorCD_t &v2)
{
    unsigned int N = v1.n_elem;
    assert(v1.n_elem == v2.n_elem);
    unsigned int inc = 1;
    return zdotu_(&N, v1.memptr(), &inc, v2.memptr(), &inc);
}

void MatrixVectorMult(const Matrix_t &A, const SiteVector_t &X, const double &alpha, SiteVector_t &Y)
{
    const unsigned int N = A.n_cols();
    assert(N == X.n_elem);
    double beta = 0.0;
    const char trans = 'n';
    const unsigned int inc = 1;
    const unsigned int ld_A = A.mem_n_rows();
    dgemv_(&trans, &N, &N, &alpha, A.memptr(), &ld_A, X.memptr(), &inc, &beta, Y.memptr(), &inc);
}

void MatrixVectorMult(const MatrixCD_t &A, const SiteVectorCD_t &X, cd_t &alpha, SiteVectorCD_t &Y)
{
    unsigned int N = A.n_cols();
    assert(N == X.n_elem);
    cd_t beta(0.0, 0.0);
    char trans = 'n';
    unsigned int inc = 1;
    const unsigned int ld_A = A.mem_n_rows();
    zgemv_(&trans, &N, &N, &alpha, A.memptr(), &ld_A, X.memptr(), &inc, &beta, Y.memptr(), &inc);
}

void VectorMatrixMult(SiteVector_t const &X, Matrix_t const &A, const double &alpha, SiteVector_t &Y)
{
    unsigned int N = A.n_cols();
    assert(N == X.n_elem);
    double beta = 0.0;
    char trans = 't';
    unsigned int inc = 1;
    const unsigned int ld_A = A.mem_n_rows();
    dgemv_(&trans, &N, &N, &alpha, A.memptr(), &ld_A, X.memptr(), &inc, &beta, Y.memptr(), &inc);
}

void VectorMatrixMult(SiteVectorCD_t const &X, MatrixCD_t const &A, const cd_t &alpha, SiteVectorCD_t &Y)
{
    unsigned int N = A.n_cols();
    assert(N == X.n_elem);
    cd_t beta(0.0, 0.0);
    char trans = 't';
    unsigned int inc = 1;
    const unsigned int ld_A = A.mem_n_rows();
    zgemv_(&trans, &N, &N, &alpha, A.memptr(), &ld_A, X.memptr(), &inc, &beta, Y.memptr(), &inc);
}

double Dot(const SiteVector_t &v1, const Matrix_t &A, const SiteVector_t &v2)
{
    const size_t N = A.n_cols();
    SiteVector_t dummy(N);
    double one = 1.0;
    MatrixVectorMult(A, v2, one, dummy);
    return DotVectors(v1, dummy);
}

std::complex<double> Dot(const SiteVectorCD_t &v1, const MatrixCD_t &A, const SiteVectorCD_t &v2)
{
    const size_t N = A.n_cols();
    SiteVectorCD_t dummy(N);
    cd_t one(1.0, 0.0);
    MatrixVectorMult(A, v2, one, dummy);
    return DotVectors(v1, dummy);
}

void DGEMM(const double &alpha, const double &beta, const Matrix_t &A,
           const Matrix_t &B, Matrix_t &C, const size_t &colNum = 0)
{
    //performs: C = alpha*A*B + beta*C
    // C is size n_rowsC x n_colsC
    assert(A.n_cols() == B.n_rows());
    assert(C.n_rows() == A.n_rows());
    assert((C.n_cols() - colNum) == B.n_cols());
    const unsigned int n_rowsC = A.n_rows();
    const unsigned int ld_A = A.mem_n_rows();
    const unsigned int n_colsC = B.n_cols();
    const unsigned int ld_B = B.mem_n_rows();
    const unsigned int n_rowsB = B.n_rows();
    const unsigned int ld_C = C.mem_n_rows();

    char no = 'n';

    const size_t elemNum = ld_C * colNum;
    dgemm_(&no, &no, &n_rowsC, &n_colsC, &n_rowsB, &alpha, A.memptr(),
           &ld_A, B.memptr(), &ld_B, &beta, &(C.memptr()[elemNum]), &ld_C);
    return;
}

Matrix_t DotRank2(const Matrix_t &m1, const Matrix_t &A, const Matrix_t &m2)
{
    //result = m1*A*m2
    Matrix_t C(A.n_rows(), m2.n_cols()); //C = A*m2
    C.Zeros();
    DGEMM(1.0, 0.0, A, m2, C);

    Matrix_t result(m1.n_rows(), C.n_cols());
    result.Zeros();
    DGEMM(1.0, 0.0, m1, C, result); //result = m1*C
    return result;
}

void TriangularSolve(const char &uplo, const char &trans, const Matrix_t &A, SiteVector_t &B)
{
    const char diag = uplo == 'l' ? 'u' : 'n'; //if lower triangular, diagonal  ones (1)

    const unsigned int N = A.n_cols();
    const unsigned int nrhs = 1;
    const unsigned int ldb = B.n_rows;
    const unsigned int ld_A = A.mem_n_rows();
    const unsigned int info = 0;
    assert(N == ldb);

    dtrtrs_(&uplo, &trans, &diag, &N, &nrhs, A.memptr(), &ld_A, B.memptr(), &ldb, &info);
    assert(info == 0);
    return;
    // return 0;
}

void ExtractLU(const Matrix_t &LU, Matrix_t &L, Matrix_t &U)
{
    const unsigned int ld_LU = LU.mem_n_rows();
    const unsigned int ld_L = L.mem_n_rows();
    const unsigned int ld_U = U.mem_n_rows();
    const unsigned int M = LU.n_rows();
    assert(M == L.n_rows());
    assert(M == U.n_rows());

    const char up = 'u';
    const char lo = 'l';
    const double a = 0.0;
    const double b = 1.0;
    dlacpy_(&lo, &M, &M, LU.memptr(), &ld_LU, L.memptr(), &ld_L);
    dlaset_(&up, &M, &M, &a, &b, L.memptr(), &ld_L);
    dlaset_(&lo, &M, &M, &a, &b, U.memptr(), &ld_U);
    dlacpy_(&up, &M, &M, LU.memptr(), &ld_LU, U.memptr(), &ld_U);
    return;
}

void TriangularInverse(const char &uplo, Matrix_t &A)
{
    const char diag = uplo == 'l' ? 'u' : 'n'; //if lower triangular, diagonal  ones (1)

    const unsigned int N = A.n_cols();
    const unsigned int ld_A = A.mem_n_rows();
    const unsigned int info = 0;
    assert(N == A.n_rows());

    dtrtri_(&uplo, &diag, &N, A.memptr(), &ld_A, &info);
    assert(info == 0);
    return;
    // return 0;
}

void LUInverse(const double &alpha, Matrix_t &L, Matrix_t &U, const unsigned int &M)
{
    const unsigned int N = U.n_rows();
    assert(N == L.n_rows());
    assert(N == M);
    const unsigned int ld_L = L.mem_n_rows();
    const unsigned int ld_U = U.mem_n_rows();

    TriangularInverse('u', U);

    //Solve for A^-1:  A^-1*L=U^-1, where A=LU  M = n_rows=n_cols
    const char side = 'r';
    const char uplo = 'l';
    const char trans = 'n';
    const char diag = 'u';
    dtrsm_(&side, &uplo, &trans, &diag, &M, &M, &alpha, L.memptr(), &ld_L, U.memptr(), &ld_U);

    return;
}

//Upgrade the matrix if the last element of the inverse is known (STilde)
void BlockRankOneUpgrade(Matrix_t &mk, const SiteVector_t &mkQ, const SiteVector_t &R, const double &STilde)
{
    //mkQ = m^{k}*Q, needed to calculate STilde, see Gull CTQMC review
    const unsigned int k = mk.n_cols();
    const unsigned int kp1 = k + 1;
    const double one = 1.0;

    SiteVector_t Rmk(k);
    VectorMatrixMult(R, mk, one, Rmk);

    SiteVector_t QTilde = -STilde * mkQ;
    SiteVector_t RTilde = -STilde * Rmk;

    const unsigned int inc = 1;
    const unsigned int ld_mk = mk.mem_n_rows();

    dger_(&k, &k, &STilde, &(mkQ.memptr()[0]), &inc, &(Rmk.memptr()[0]), &inc, mk.memptr(), &ld_mk);

    mk.Resize(kp1, kp1);
    const unsigned int ld_mk_resized = mk.mem_n_rows();
    dcopy_(&k, &(RTilde.memptr()[0]), &inc, &(mk.memptr()[k]), &ld_mk_resized);
    dcopy_(&k, &(QTilde.memptr()[0]), &inc, &(mk.memptr()[ld_mk_resized * k]), &inc);

    mk(k, k) = STilde;
    return;
}

// Upgrade the matrix if the last matrix element of the inverse is known (STilde)
void BlockRankTwoUpgrade(Matrix_t &mk, const Matrix_t &Q, const Matrix_t &R, const Matrix_t &STilde)
{

    const unsigned int k = mk.n_cols();
    const unsigned int kp2 = k + 2;
    const double one = 1.0;
    const double zero = 0.0;

    Matrix_t mkQ(mk.n_rows(), Q.n_cols());
    mkQ.Zeros();
    Matrix_t Rmk(R.n_rows(), mk.n_cols());
    Rmk.Zeros();

    DGEMM(one, zero, mk, Q, mkQ);
    DGEMM(one, zero, R, mk, Rmk);

    Matrix_t QTilde(mkQ.n_rows(), STilde.n_cols());
    Matrix_t RTilde(STilde.n_rows(), Rmk.n_cols());

    DGEMM(-one, zero, mkQ, STilde, QTilde);
    DGEMM(-one, zero, STilde, Rmk, RTilde);

    mk += DotRank2(mkQ, STilde, Rmk);
    mk.Resize(kp2, kp2);

    // const char all = 'A';

    // //copy QTIlde in mk
    // // dlacpy_(&all, &QTilde.n_rows(), &QTilde.n_cols(), QTilde.memptr(), &QTilde.mem_n_rows(), mk.memptr()[k], &mk.mem_n_rows());

    // //copy RTilde in mk
    // dcopy_(&k, RTilde.memptr(), &inc, &(mk.memptr()[k]), &ld_mk_resized);
    // dcopy_(&k, &(QTilde.memptr()[0]), &inc, &(mk.memptr()[ld_mk_resized * k]), &inc);
    // // dlacpy_(&all, &RTilde.n_rows(), &RTilde.n_cols(), RTilde.memptr(), &RTilde.mem_n_rows(), mk.memptr()[start], &mk.mem_n_cols());
    for (size_t ii = 0; ii < k; ii++)
    {
        mk(ii, k) = QTilde(ii, 0);
        mk(ii, k + 1) = QTilde(ii, 1);

        mk(k, ii) = RTilde(0, ii);
        mk(k + 1, ii) = RTilde(1, ii);
    }

    mk(k, k) = STilde(0, 0);
    mk(k, k + 1) = STilde(0, 1);
    mk(k + 1, k) = STilde(1, 0);
    mk(k + 1, k + 1) = STilde(1, 1);

    return;
}

//pp row and col to remove
void BlockRankOneDowngrade(Matrix_t &m1, const size_t &pp)
{

    const unsigned int inc = 1;
    const unsigned int kk = m1.n_rows();
    const unsigned int kkm1 = kk - 1;
    const unsigned int ld_m1 = m1.mem_n_rows();

    if (kkm1 == 0)
    {
        m1.Clear();
    }
    else
    {
        m1.SwapRows(pp, kkm1);
        m1.SwapCols(pp, kkm1);
        SiteVector_t lastRow;
        SiteVector_t lastCol;
        ExtractRow(kkm1, lastRow, m1);
        ExtractCol(kkm1, lastCol, m1);
        const double alpha = -1.0 / m1(kkm1, kkm1);
        // this next line does S = S - A12 A22^(-1) A21;
        dger_(&kkm1, &kkm1, &alpha, lastCol.memptr(), &inc, lastRow.memptr(), &inc, m1.memptr(), &ld_m1);
        m1.Resize(kkm1, kkm1);
    }
}

//pp row and col to remove
void BlockDowngrade(Matrix_t &m1, const size_t &pp, const size_t &nn)
{
    //pp = row and col number to start remove
    //nn = number of rows and col to remove
    // const unsigned int inc = 1;
    const unsigned int kk = m1.n_rows();
    assert(kk >= nn);
    const unsigned int kkmnn = kk - nn;
    // const unsigned int ld_m1 = m1.mem_n_rows();

    if (kkmnn == 0)
    {
        m1.Clear();
    }
    else
    {
        for (size_t ii = 0; ii < nn; ii++)
        {
            const size_t cc = nn - 1 - ii;
            m1.SwapRowsAndCols(pp + cc, kk - 1 - ii);
        }

        Matrix_t B; //right-upper block of m1, size = kkmnn x nn
        Matrix_t C; //left-lower block of m1, size = nn x kkmnn
        Matrix_t D; //lower-right block of m1, size = nn x nn
        B = GetSubMat(0, kkmnn, kkmnn, kk, m1);
        C = GetSubMat(kkmnn, 0, kk, kkmnn, m1);
        D = GetSubMat(kkmnn, kkmnn, kk, kk, m1);

        std::cout << "B.n_rows = " << B.n_rows() << std::endl;
        D.Inverse();
        std::cout << " D.n_rows() = " << D.n_rows() << std::endl;
        std::cout << " kkmnn = " << kkmnn << std::endl;

        Matrix_t DInverseC(nn, kkmnn);
        DInverseC.Zeros();

        DGEMM(1.0, 0.0, D, C, DInverseC);

        m1.Resize(kkmnn, kkmnn);
        DGEMM(-1.0, 1.0, B, DInverseC, m1);
    }
}

//double-diagonal matrix-general matrix multiplication
//B = diag*A
void DDMGMM(const SiteVector_t &diag, const Matrix_t &A, Matrix_t &B)
{
    const unsigned int diag_size = diag.n_elem;
    assert(diag_size == A.n_rows());
    assert(diag_size == A.n_cols());

    B = A;
    const unsigned int ld_B = B.mem_n_rows();
    // const unsigned int inc = 1;
    for (unsigned int i = 0; i < diag_size; i++)
    {
        double alpha = diag(i);
        dscal_(&diag_size, &alpha, &(B.memptr()[i]), &ld_B);
    }
}

//double-diagonal matrix-general matrix multiplication
//A = diag*A
void DDMGMM(const SiteVector_t &diag, Matrix_t &A)
{
    const unsigned int diag_size = diag.n_elem;
    assert(diag_size == A.n_rows());
    assert(diag_size == A.n_cols());

    const unsigned int ld_A = A.mem_n_rows();
    // const unsigned int inc = 1;
    for (unsigned int i = 0; i < diag_size; i++)
    {
        double alpha = diag(i);
        dscal_(&diag_size, &alpha, &(A.memptr()[i]), &ld_A);
    }
}

} // namespace LinAlg
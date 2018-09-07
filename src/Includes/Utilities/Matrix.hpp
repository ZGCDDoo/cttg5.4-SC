#pragma once

#include "Utilities.hpp"

namespace LinAlg
{

extern "C"
{
    //Vector-vector and vector-matrix operations
    unsigned int dger_(unsigned int const *, unsigned int const *, double const *, double const *, unsigned int const *, double const *, unsigned int const *, double *, unsigned int const *);
    double ddot_(unsigned int const *, double const *, unsigned int const *, double const *, unsigned int const *);
    double _Complex zdotu_(unsigned int const *, cd_t const *, unsigned int const *, cd_t const *, unsigned int const *);
    unsigned int dgemv_(char const *, unsigned int const *, unsigned int const *, double const *, double const *, unsigned int const *, double const *, unsigned int const *, double const *, double *, unsigned int const *);
    unsigned int zgemv_(char const *, unsigned int const *, unsigned int const *, cd_t const *, cd_t const *, unsigned int const *, cd_t const *, unsigned int const *, cd_t const *, cd_t *, unsigned int const *);
    unsigned int dcopy_(unsigned int const *, double const *, unsigned int const *, double *, unsigned int const *);
    unsigned int dscal_(unsigned int const *, double const *, double *, unsigned int const *);

    //Matrix operations
    unsigned int dlacpy_(char const *, unsigned int const *, unsigned int const *, double const *, unsigned int const *, double *, unsigned int const *);
    unsigned int slacpy_(char const *, unsigned int const *, unsigned int const *, double const *, unsigned int const *, double *, unsigned int const *);
    unsigned int dlaset_(char const *, unsigned int const *, unsigned int const *, double const *, double const *, double *, unsigned int const *);
    unsigned int dgemm_(char const *, char const *, unsigned int const *, unsigned int const *, unsigned int const *,
                        double const *, double const *, unsigned int const *, double const *,
                        unsigned int const *, double const *, double *, unsigned int const *);

    unsigned int dtrtrs_(char const *, char const *, char const *, unsigned int const *, unsigned int const *, double const *,
                         unsigned int const *, double *, unsigned int const *, unsigned int const *);

    unsigned int dtrsm_(char const *, char const *, char const *, char const *, unsigned int const *, unsigned int const *,
                        double const *, double const *, unsigned int const *, double *, unsigned int const *);

    unsigned int dtrtri_(char const *, char const *, unsigned int const *, double *,
                         unsigned int const *, unsigned int const *);
    void dgesv_(const unsigned int *, const unsigned int *, double *, const unsigned int *, unsigned int *, double *, const unsigned int *, int *);
}

template <typename T>
class Matrix
{
  public:
    static const size_t INIT_SIZE;

    Matrix() : n_rows_(0), n_cols_(0), mat_(INIT_SIZE, INIT_SIZE){};
    Matrix(const size_t &n_rows, const size_t &n_cols) : n_rows_(n_rows), n_cols_(n_cols), mat_(n_rows, n_cols){};

    Matrix(const arma::Mat<T> &m1)
    {
        n_rows_ = m1.n_rows;
        n_cols_ = m1.n_cols;
        mat_ = m1;
    }

    Matrix(const arma::Mat<double> &m1, const arma::Mat<double> &m2);

    inline const Matrix<T> &operator=(const Matrix<T> &m1)
    {
        n_rows_ = m1.n_rows_;
        n_cols_ = m1.n_cols_;
        mat_ = m1.mat_;
        return *this;
    }

    Matrix(std::initializer_list<std::initializer_list<T>> initListofLists) : Matrix(arma::Mat<T>(initListofLists)) {}

    static Matrix<T> DiagMat(const arma::Col<T> &v1)
    {
        return Matrix<T>(arma::diagmat(v1));
    }

    static Matrix<T> DiagMat(const size_t &size, const T &value)
    {
        Matrix<T> tmp(size, size);
        tmp.Zeros();
        for (size_t i = 0; i < size; i++)
        {
            tmp(i, i) = value;
        }
        return tmp;
    }

    void AssertSizes(const size_t &i, const size_t &j) const
    {
        assert(i <= n_rows_);
        assert(j <= n_cols_);
    }

    inline T &operator()(const size_t &i, const size_t &j)
    {
        AssertSizes(i, j);
        return mat_(i, j);
    }

    inline const T &operator()(const size_t &i, const size_t &j) const
    {
        AssertSizes(i, j);
        return mat_(i, j);
    }

    inline size_t n_rows() const
    {
        return n_rows_;
    }

    inline size_t n_cols() const
    {
        return n_cols_;
    }

    arma::Mat<T> mat() const
    {
        return mat_;
    }

    inline size_t mem_n_rows() const
    {

        return mat_.n_rows;
    }

    inline size_t mem_n_cols() const
    {

        return mat_.n_cols;
    }

    T *memptr()
    {
        return mat_.memptr();
    }

    const T *memptr() const
    {
        const T *const mem_out = mat_.memptr();
        return mem_out;
    }

    inline void Resize(const size_t &n_rows, const size_t &n_cols)
    {

        if (n_rows > mem_n_rows() || n_cols > mem_n_cols())
        {
            //Resize by ~20 % test this
            mat_.resize(1.20 * (n_rows + 1), 1.20 * (n_cols + 1));
        }

        n_rows_ = n_rows;
        n_cols_ = n_cols;
    }

    void SetSize(const size_t &n_rows, const size_t &n_cols)
    {

        if (n_rows > mem_n_rows() || n_cols > mem_n_cols())
        {
            //Resize by ~20 % test this
            mat_.set_size(1.20 * (n_rows + 1), 1.20 * (n_cols + 1));
        }

        n_rows_ = n_rows;
        n_cols_ = n_cols;
    }

    Matrix<T> Transpose()
    {
        Matrix tmp = *this;
        tmp.mat_.t();
        return tmp;
    }

    void CopyVectorInCol(arma::Col<T> &col, const size_t &p);

    void CopyVectorInRow(arma::Col<T> &row, const size_t &p);

    void SwapRows(const size_t &r1, const size_t &r2)
    {
        AssertSizes(r1, r2);
        //Change this to use dswap_ of blas, not a priority
        mat_.swap_rows(r1, r2);
    }

    void SwapCols(const size_t &c1, const size_t &c2)
    {
        AssertSizes(c1, c2);
        //Change this to use dswap_ of blas, not a priority
        mat_.swap_cols(c1, c2);
    }

    void SwapRowsAndCols(const size_t &c1, const size_t &c2)
    {
        AssertSizes(c1, c2);
        //Change this to use dswap_ of blas, not a priority
        mat_.swap_cols(c1, c2);
        mat_.swap_rows(c1, c2);
    }

    void Zeros()
    {
        mat_.zeros();
    }

    void Ones()
    {
        mat_.ones();
    }

    void Eye()
    {
        mat_.eye();
    }

    void Clear(const size_t &n_rows = 0, const size_t &n_cols = 0)
    {
        SetSize(n_rows, n_cols);
        n_rows_ = n_cols_ = 0;
    }

    void Swap(Matrix<T> &dummy)
    {
        mat_.swap(dummy.mat_);
        size_t tmp_row = n_rows_;
        size_t tmp_col = n_cols_;
        n_cols_ = dummy.n_cols_;
        n_rows_ = dummy.n_rows_;
        dummy.n_cols_ = tmp_col;
        dummy.n_rows_ = tmp_row;
    }

    void MultCol(const size_t &j, const double &val)
    {
        mat_.col(j) *= val;
    }

    void SubMat(const size_t &r1, const size_t &c1, const size_t &r2, const size_t c2, Matrix<T> mIn)
    {
        //copy all of the matrix  mIn to the current matrix
        mIn.mat_.resize(mIn.n_rows_, mIn.n_cols_);
        mat_.submat(r1, c1, r2, c2) = mIn.mat_;
    }

    void SubMat(const size_t &r1, const size_t &c1, const size_t &r2, const size_t c2, const double &val)
    {
        const char all = 'A';
        const unsigned int M = r2 - r1 + 1;
        const unsigned int N = c2 - c1 + 1;
        const unsigned int ld_mat = mem_n_rows();

        dlaset_(&all, &M, &N, &val, &val, &(mat_.memptr()[r1 + c1 * mem_n_rows()]), &ld_mat);
    }

    void ShedRowAndCol(const size_t &r)
    {
        mat_.shed_row(r);
        mat_.shed_col(r);
        n_rows_--;
        n_cols_--;
    }
    //addition d'une matrice
    template <typename S>
    Matrix<T> &operator+=(const Matrix<S> &A)
    {
        assert(A.n_rows() == n_rows() && A.n_cols() == n_cols());

        for (size_t jj = 0; jj < n_cols(); jj++)
        {
            for (size_t ii = 0; ii < n_rows(); ii++)
            {
                mat_(ii, jj) += A(ii, jj);
            }
        }
        return *this;
    }

    //soustraction d'une matrice
    template <typename S>
    Matrix<T> &operator-=(const Matrix<S> &A)
    {
        mat_ -= A.mat_;
        return *this;
    }

    //multiplication par un scalaire
    inline Matrix<T> operator*=(const T &a)
    {
        mat_ *= a;
        return *this;
    }

    void Load(const std::string &fname)
    {
        mat_.load(fname);
        n_rows_ = mat_.n_rows;
        n_cols_ = mat_.n_cols;
    }

    // void Save(const std::string &fname) const
    // {
    //     arma::Mat<T> tmp = mat_.submat(0, 0, n_rows, n_cols);
    //     tmp.save(fname, arma::raw_ascii);
    // }

    void Print() const
    {
        mat_.print();
    }

    void Inverse();

    T Determinant()
    {
        mat_.resize(n_rows_, n_cols_);
        return arma::det(mat_);
    }

  private:
    size_t n_rows_; //the real size of the matrix
    size_t n_cols_;
    arma::Mat<T> mat_; // a matrix bigger than neccessary
};

template <typename T>
const size_t Matrix<T>::INIT_SIZE = 16;

template <>
Matrix<cd_t>::Matrix(const arma::Mat<double> &m1, const arma::Mat<double> &m2)
{
    n_rows_ = m1.n_rows;
    n_cols_ = m1.n_cols;
    mat_ = arma::Mat<cd_t>(m1, m2);
}
// addition(opérateur binaire)
template <typename T>
inline Matrix<T> operator+(const Matrix<T> &x, const Matrix<T> &y)
{
    Matrix<T> tmp(x);
    tmp += y;
    return tmp;
}

//soustraction (opérateur binaire)
template <typename T>
inline Matrix<T> operator-(const Matrix<T> &x, const Matrix<T> &y)
{
    Matrix<T> tmp(x);
    tmp -= y;
    return tmp;
}

//multiplication par un scalaire (opérateur binaire)
template <typename T>
inline Matrix<T> operator*(const Matrix<T> &x, const T &a)
{
    Matrix<T> tmp(x);
    tmp *= a;
    return tmp;
}

template <typename T>
inline Matrix<T> operator*(const T &a, const Matrix<T> &x)
{
    Matrix<T> tmp(x);
    tmp *= a;
    return tmp;
}

template <>
void Matrix<double>::CopyVectorInCol(arma::Col<double> &col, const size_t &p)
{
    assert(col.n_elem == n_rows_);
    assert(p <= n_cols_);

    const unsigned int inc = 1;
    const unsigned int k = n_rows_;
    const unsigned int mem_k = mem_n_rows();
    dcopy_(&k, &(col.memptr()[0]), &inc, &(memptr()[mem_k * p]), &inc);
}

template <>
void Matrix<double>::CopyVectorInRow(arma::Col<double> &row, const size_t &p)
{
    assert(row.n_elem == n_cols_);
    assert(p <= n_rows_);

    const unsigned int inc = 1;
    const unsigned int k = n_cols_;
    const unsigned int mem_k = mem_n_rows();
    dcopy_(&k, &(row.memptr()[0]), &inc, &(memptr()[p]), &mem_k);
}

template <>
void Matrix<double>::Inverse()
{
    assert(n_rows_ == n_cols_);
    Matrix<double> tmp = *this;
    *this = DiagMat(n_rows_, 1.0);

    const unsigned int dim = n_rows_;
    const unsigned int ld_tmp = tmp.mem_n_rows();
    const unsigned int ld_this = mem_n_rows();
    std::vector<unsigned int> ipiv(dim);
    int info;
    dgesv_(&dim, &dim, tmp.memptr(), &ld_tmp, ipiv.data(), memptr(), &ld_this, &info);
    assert(info == 0);

    // return *this;
}

} //namespace LinAlg
#pragma once

#include "Utilities.hpp"
#include "IO.hpp"

namespace TransformSquare2x2
{

const size_t Nc = 4;
const size_t N_K = 3; //number of indepedant big K

struct Transform
{
    //Real_Sites: 0, 1, 2,3
    //KSites: 00, 0Pi, PiPi, 00

    Transform() : U_(Nc, Nc)
    {
        U_(0, 0) = 1.;
        U_(0, 1) = 1.;
        U_(0, 3) = 1.;
        U_(0, 2) = 1.;

        U_(1, 0) = 1.;
        U_(1, 1) = 1.;
        U_(1, 3) = -1.;
        U_(1, 2) = -1.;

        U_(3, 0) = 1.;
        U_(3, 1) = -1.;
        U_(3, 3) = -1.;
        U_(3, 2) = 1.;

        U_(2, 0) = 1.;
        U_(2, 1) = -1.;
        U_(2, 3) = 1.;
        U_(2, 2) = -1.;

        U_ /= 2.;
    };

    ClusterCubeCD_t KtoR(const ClusterCubeCD_t &k)
    {
        assert(k.n_rows == k.n_cols);
        assert(k.n_rows == Nc);

        ClusterCubeCD_t r = k;
        ClusterMatrixCD_t temp = k.slice(0);

        for (size_t i = 0; i < k.n_slices; i++)
        {
            temp = U_.t() * k.slice(i);
            r.slice(i) = temp * U_;
        }
        return r;
    }

    ClusterCubeCD_t RtoK(const ClusterCubeCD_t &r)
    {
        assert(r.n_rows == r.n_cols);
        assert(r.n_rows == Nc);
        ClusterCubeCD_t k = r;
        ClusterMatrixCD_t temp = r.slice(0);

        for (size_t i = 0; i < r.n_slices; i++)
        {
            temp = U_ * r.slice(i);
            k.slice(i) = temp * U_.t();
        }
        return k;
    }

  private:
    ClusterMatrixCD_t U_;
};

void RtoK(const ClusterCubeCD_t &r, const double &beta, const size_t &iteration, const std::string &name)
{
    std::cout << "STart RtoK" << std::endl;

    Transform transform;
    std::cout << "After define transform" << std::endl;
    ClusterCubeCD_t k = transform.RtoK(r);

    std::cout << "After Transform" << std::endl;

    assert(k.n_rows == Nc);
    assert(k.n_cols == r.n_rows);

    ClusterMatrix_t matOut(k.n_slices, 2 * N_K + 1);

    std::cout << "Before loop" << std::endl;
    for (size_t n = 0; n < k.n_slices; n++)
    {
        matOut(n, 0) = (2.0 * n + 1.0) * M_PI / beta;

        matOut(n, 1) = k(0, 0, n).real();
        matOut(n, 2) = k(0, 0, n).imag();

        matOut(n, 3) = k(1, 1, n).real();
        matOut(n, 4) = k(1, 1, n).imag();

        matOut(n, 5) = k(2, 2, n).real();
        matOut(n, 6) = k(2, 2, n).imag();
    }
    std::cout << "After loop" << std::endl;
    std::string filename = name + "_K" + std::to_string(iteration) + ".dat";
    matOut.save(filename, arma::raw_ascii);

    std::cout << "End RtoK" << std::endl;
    return;
}
}
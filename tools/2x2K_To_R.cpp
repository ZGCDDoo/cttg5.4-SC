
#include <cmath>
#include <iostream>
#include <fstream>
#include <valarray>
#include <vector>
#include <utility>
#include <set>
#include <algorithm>
#include <fstream>
#include <cassert>
#include <vector>
#include <string>
#include <ccomplex>

//External Libraries
#include <armadillo>
#include "../src/Includes/Utilities/TransformSquare2x2.hpp"
#include "../src/Includes/Utilities/IO.hpp"

//Inspired by Patrick Sémon

using SiteVectorCD_t = arma::cx_vec;
using SiteRowCD_t = arma::cx_rowvec;
using ClusterSitesCD_t = std::vector<arma::cx_vec>;
using ClusterMatrixCD_t = arma::cx_mat;
using ClusterCubeCD_t = arma::cx_cube;

using SiteVector_t = arma::vec;
using SiteRow_t = arma::rowvec;
using ClusterSites_t = std::vector<arma::vec>;
using ClusterMatrix_t = arma::mat;
using ClusterCube_t = arma::cube;

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        throw std::runtime_error("Miseria: Wrong number of input parameters. Stupido !");
    }

    const std::string fname_hyb = argv[1];
    const double BETA = atof(argv[2]);
    std::cout << "BETA = " << BETA << std::endl;
    std::cout << "fname = " << fname_hyb << std::endl;
    //0.) Read Patricks hyb.dat
    ClusterMatrix_t hybDatIrr;
    hybDatIrr.load(fname_hyb);
    hybDatIrr.shed_col(0); //we dont want the matsubara frequencies;

    //1.) Convert it to ClusterCube
    const size_t Nc = 4;
    ClusterCubeCD_t hybIrr(Nc, Nc, hybDatIrr.n_rows);
    hybIrr.zeros();
    for (size_t i = 0; i < hybIrr.n_slices; i++)
    {
        ClusterMatrixCD_t tmp(Nc, Nc);
        tmp.zeros();
        tmp(0, 0) = std::complex<double>(hybDatIrr(i, 0), hybDatIrr(i, 1));
        tmp(1, 1) = std::complex<double>(hybDatIrr(i, 2), hybDatIrr(i, 3));
        tmp(3, 3) = tmp(1, 1); //std::complex<double>(hybDatIrr(i, 2), hybDatIrr(i, 3));
        tmp(2, 2) = std::complex<double>(hybDatIrr(i, 4), hybDatIrr(i, 5));
        hybIrr.slice(i) = tmp;
    }

    //2.)Convert it to hybReal;
    TransformSquare2x2::Transform transform;
    ClusterCubeCD_t hybReal = transform.KtoR(hybIrr);

    //3.)convert it to .arma - CD matrix
    using IOModel_t = IO::IOSquare2x2;
    IOModel_t ioModel;

    //the file hybCD-out.arma is ok, but not the hybCD-out.dat, which is garbage.
    ioModel.SaveCube("2x2K_to_R_converted", hybReal, BETA, 10, true);

    return 0;
}
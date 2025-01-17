#pragma once

#include <cstdlib>
#include <armadillo>
#include "Utilities.hpp"
#include "../IS/ISResult.hpp"

#ifdef HAVEMPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

typedef std::map<std::string, std::valarray<size_t>> UpdStats_t;

namespace mpiUt
{

const int master = 0;
int NWorkers()
{
#ifdef HAVEMPI
    mpi::communicator world;
    return world.size();
#endif
#ifndef HAVEMPI
    return 1;
#endif
}

int Rank()
{
#ifdef HAVEMPI
    mpi::communicator world;
    return world.rank();
#endif
#ifndef HAVEMPI
    return 0;
#endif
}

void Print(const std::string &message)
{
#ifdef HAVEMPI
    mpi::communicator world;
    if (world.rank() == 0)
    {
        std::cout << "Rank " << std::to_string(Rank()) << ": " << message << std::endl;
    }
#endif

#ifndef HAVEMPI
    std::cout << message << std::endl;
#endif
}

template <typename TIOModel>
class IOResult
{
  public:
    static void SaveISResults(const std::vector<Result::ISResult> &isResultVec, const double &beta)
    {
        TIOModel ioModel;
        assert(NWorkers() == static_cast<int>(isResultVec.size()));

        std::valarray<cd_t> nambuResult = isResultVec.at(0).nambu_;
        std::valarray<double> fillingResultUp = isResultVec.at(0).fillingUp_;
        std::valarray<double> fillingResultDown = isResultVec.at(0).fillingDown_;

        //Average the greens of matsubara, and the fillingsSigma_.
        for (int i = 1; i < NWorkers(); i++)
        {
            nambuResult += isResultVec.at(i).nambu_;

            fillingResultUp += isResultVec.at(i).fillingUp_;
            fillingResultDown += isResultVec.at(i).fillingDown_;
        }

        nambuResult /= static_cast<double>(NWorkers());
        fillingResultUp /= static_cast<double>(NWorkers());
        fillingResultDown /= static_cast<double>(NWorkers());

        //convert the greens to ClusterMatrixCD_t
        const size_t n_rows = isResultVec.at(0).n_rows_;
        const size_t n_cols = isResultVec.at(0).n_cols_;
        const size_t n_slices = isResultVec.at(0).n_slices_;

        const ClusterCubeCD_t nambuResultCube = Utilities::VecCDToCubeCD<std::valarray<cd_t>>(nambuResult, n_rows, n_cols, n_slices);
        ioModel.SaveCube("green", nambuResultCube, beta);

        //Average the obsScale_
        SaveFillingMatrixs(fillingResultUp, fillingResultDown);
        StatsJsons(isResultVec);
    }

    static void StatsJsons(const std::vector<Result::ISResult> &isResultVec)
    {
        Json jjResult(isResultVec.at(0).obsScal_);

        const size_t nworkers = NWorkers();
        const size_t jjSize = jjResult.size();

        //START STATS===================================================================
        ClusterMatrix_t statsMat(nworkers, jjSize); //Each row contains the data for each jj
        std::vector<std::string> keys;

        for (Json::iterator it = jjResult.begin(); it != jjResult.end(); ++it)
        {
            keys.push_back(it.key());
        }

        for (size_t i = 0; i < nworkers; i++)
        {
            std::map<std::string, double> tmpMap = isResultVec.at(i).obsScal_;
            for (size_t j = 0; j < jjSize; j++)
            {
                statsMat(i, j) = tmpMap[keys.at(j)];
            }
        }

        ClusterMatrix_t stddevs = arma::stddev(statsMat, 1, 0); // size = 1 x statsMat.n_cols
        ClusterMatrix_t means = arma::mean(statsMat, 0);

        //END STATS===================================================================

        jjResult.clear();
        double sqrtNWorkers = std::sqrt(double(NWorkers()));
        for (size_t i = 0; i < keys.size(); i++)
        {
            jjResult[keys.at(i)] = {means(0, i), stddevs(0, i) / sqrtNWorkers}; // the ith key contains a vector of size 2, mean and stddev
        }

        jjResult["NWorkers"] = {NWorkers(), 0.0};

        std::ofstream fout("Obs.json");
        fout << std::setw(4) << jjResult << std::endl;
        fout.close();
    }

    static void SaveFillingMatrixs(std::valarray<double> fillingResultUp, std::valarray<double> fillingResultDown)
    {
        std::cout << "Just before saving NMatrix " << std::endl;
        TIOModel ioModel;
        ClusterMatrix_t nUpMatrix(ioModel.Nc, ioModel.Nc);
        nUpMatrix.zeros();
        ClusterMatrix_t nDownMatrix = nUpMatrix;

        for (size_t kk = 0; kk < ioModel.fillingSites().size(); kk++)
        {

            for (size_t ii = 0; ii < nUpMatrix.n_rows; ii++)
            {
                std::pair<size_t, size_t> fSites = ioModel.indepSites().at(ioModel.FindIndepSiteIndex(ii, ii));
                assert(fSites.first == fSites.second);
                auto fillingSites = ioModel.fillingSites();
                const size_t fsitesfirst = fSites.first;
                auto fIt = std::find(fillingSites.begin(), fillingSites.end(), fsitesfirst);

                if (fIt == fillingSites.end())
                {
                    throw std::runtime_error("Bad index in fIt nMatrix, SaveFillingMatrix");
                }

                size_t fsite = std::distance(fillingSites.begin(), fIt);
                nUpMatrix(ii, ii) = fillingResultUp[fsite];
                nDownMatrix(ii, ii) = fillingResultDown[fsite];
            }
        }
        assert(nUpMatrix.save("nUpMatrix.dat", arma::raw_ascii));
        assert(nDownMatrix.save("nDownMatrix.dat", arma::raw_ascii));
    }
};

void SaveUpdStats(const std::string &fname, std::vector<UpdStats_t> &updStatsVec)
{

    std::vector<std::string> keys;
    UpdStats_t updStatsResult(updStatsVec.at(0));

    for (UpdStats_t::iterator it = updStatsResult.begin(); it != updStatsResult.end(); ++it)
    {
        keys.push_back(it->first);
    }

    for (size_t i = 1; i < updStatsVec.size(); i++)
    {
        for (const std::string key : keys)
        {
            std::valarray<size_t> tmp = updStatsVec.at(i)[key];
            updStatsResult[key] += tmp;
        }
    }

    Json jjout(updStatsResult);
    for (const std::string key : keys)
    {

        updStatsResult[key][0] = double(updStatsResult[key][0]) / double(NWorkers());
        updStatsResult[key][1] = double(updStatsResult[key][1]) / double(NWorkers());
        size_t nbProposed = updStatsResult[key][0];
        size_t nbAccepted = updStatsResult[key][1];

        jjout[key] = {{"Proposed", nbProposed}, {"Accepted", nbAccepted}};
    }

    std::ofstream fout(fname + ".json");
    fout << std::setw(4) << jjout << std::endl;
    fout.close();
}

void SaveConfig(const std::vector<Utilities::Vertex> &vertices)
{
    using Utilities::Vertex;

    size_t KK = vertices.size();
    ClusterMatrix_t config(KK, 3);
    for (size_t i = 0; i < KK; i++)
    {
        Vertex vertex = vertices.at(i);
        config(i, 0) = vertex.tau();
        config(i, 1) = vertex.site();
        config(i, 2) = vertex.Ising();
    }

    std::string filename = std::string("config") + std::to_string(mpiUt::Rank()) + std::string(".dat");
    config.save(filename);
}

bool LoadConfig(std::vector<Utilities::Vertex> &vertices)
{
    using Utilities::Vertex;
    ClusterMatrix_t config;
    std::string filename = std::string("config") + std::to_string(mpiUt::Rank()) + std::string(".dat");
    if (!config.load(filename))
    {
        return false;
    }

    const double eps = 0.1;
    for (size_t i = 0; i < config.n_rows; i++)
    {
        Tau_t tau = config(i, 0);
        Site_t site = config(i, 1) + eps;
        AuxSpin_t aux = config(i, 2) > 0.0 ? AuxSpin_t::Up : AuxSpin_t::Down;
        vertices.push_back(Vertex(tau, site, aux));
    }
    // std::cout << "Config = " << std::endl;
    //config.print();
    return true;
}

} //namespace mpiUt
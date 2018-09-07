#pragma once

#include "json.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <cstdlib>
#include <ctime>
#include "Utilities.hpp"

namespace IO
{
namespace FS
{

void WriteToFile(const size_t &iter, double &value, const std::string &name)
{
    std::string fname = name + std::string(".dat");
    std::ofstream fout(fname, std::ios_base::out | std::ios_base::app);
    fout << iter << " " << value << std::endl;
    fout.close();
}

void WriteToFile(const size_t &iter, const std::vector<double> &stats, const std::string &name)
{
    assert(stats.size() == 2); //mean and stddev
    std::string fname = name + std::string(".dat");
    std::ofstream fout(fname, std::ios_base::out | std::ios_base::app);
    fout << iter << " " << stats[0] << " " << stats[1] << std::endl;
    fout.close();
}

size_t CalculateNextSeed()
{
    std::srand(std::time(nullptr));
    for (int i = 0; i < std::rand() % 100; i++)
    {
        std::rand();
    }

    Utilities::EngineTypeFibonacci3217_t randFib(std::rand());
    Utilities::UniformRngFibonacci3217_t urandFib(randFib, Utilities::UniformDistribution_t(0.0, 1.0));

    for (int i = 0; i < std::rand() % 100; i++)
    {
        urandFib();
    }

    double nextSeed = urandFib() * double(std::rand());
    return (static_cast<size_t>(nextSeed));
}

void PrepareNextIter(const std::string paramsName, const size_t &iter)
{
    // std::cout << "Start of PrepareNexIter " << std::endl;
    using boost::filesystem::copy_file;
    const std::string ext = ".dat";

    copy_file("hybNextUp.dat", std::string("hybUp") + std::to_string(iter + 1) + ext);
    copy_file("selfUp.dat", std::string("selfUp") + std::to_string(iter) + ext);
    copy_file("greenUp.dat", std::string("greenUp") + std::to_string(iter) + ext);
    copy_file("Obs.json", std::string("Obs") + std::to_string(iter) + ".json");
    copy_file("upd.meas.json", std::string("upd.meas") + std::to_string(iter) + ".json");
    copy_file("gtau.dat", std::string("g0tau") + std::to_string(iter) + ext);

#ifdef AFM
    copy_file("hybNextDown.dat", std::string("hybDown") + std::to_string(iter + 1) + ext);
    copy_file("selfDown.dat", std::string("selfDown") + std::to_string(iter) + ext);
    copy_file("greenDown.dat", std::string("greenDown") + std::to_string(iter) + ext);
#endif

    std::string fname = paramsName + std::to_string(iter) + std::string(".json");
    std::ifstream fin(fname);
    Json params;
    fin >> params;
    fin.close();

    Json results;
    fin.open("Obs.json");
    fin >> results;
    fin.close();

    for (Json::iterator it = results.begin(); it != results.end(); ++it)
    {
        std::vector<double> stats = it.value();
        WriteToFile(iter, stats, it.key());
    }

    size_t nextSeed = CalculateNextSeed();
    params["SEED"] = static_cast<size_t>(nextSeed);

    params["HybFileUp"] = std::string("hybUp") + std::to_string(iter + 1);

#ifdef AFM
    params["HybFileDown"] = std::string("hybDown") + std::to_string(iter + 1);
#endif

    if (params.find("n") != params.end())
    {
        double nParams = params["n"];
        double nResult = results["n"].at(0); //mean of n from simulation
        params["mu"] = double(params["mu"]) - double(params["S"]) * (nResult - nParams);
    }
    std::ofstream fout(std::string("params") + std::to_string(iter + 1) + std::string(".json"));
    fout << std::setw(4) << params << std::endl;
    fout.close();

    return;
}

} //namespace IO
} //namespace FS
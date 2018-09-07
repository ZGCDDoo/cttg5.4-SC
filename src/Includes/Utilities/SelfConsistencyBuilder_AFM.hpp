#pragma once

#include "SelfConsistency.hpp"

namespace SelfCon
{

std::unique_ptr<ABC_SelfConsistency> SelfConsistencyBuilder_AFM(const Json &jj, const FermionSpin_t &spin)
{
    const std::string modelType = jj["modelType"].get<std::string>();

    if (modelType == "Square2x2_AFM")
    {
        const size_t Nx = 2;
        using Model_t = Models::ModelSquare2x2_AFM;
        using IOModel_t = IO::IOSquare2x2_AFM;
        using H0_t = Models::H0Square<Nx, Nx>;

        Model_t model(jj);
        IOModel_t ioModel;

        ClusterCubeCD_t greenImpurity;
        if (spin == FermionSpin_t::Up)
        {
            greenImpurity = ioModel.ReadGreenDat("greenUp.dat");
        }
        else if (spin == FermionSpin_t::Down)
        {
            greenImpurity = ioModel.ReadGreenDat("greenDown.dat");
        }

        using SelfCon_t = SelfCon::SelfConsistency<IOModel_t, Model_t, H0_t>;
        return std::make_unique<SelfCon_t>(SelfCon_t(jj, model, greenImpurity, spin));
    }
    else if (modelType == "Triangle2x2_AFM")
    {
        const size_t Nx = 2;
        using Model_t = Models::ModelTriangle2x2;
        using IOModel_t = IO::IOTriangle2x2;
        using H0_t = Models::H0Square<Nx, Nx>;

        Model_t model(jj);
        IOModel_t ioModel;

        ClusterCubeCD_t greenImpurity;
        if (spin == FermionSpin_t::Up)
        {
            greenImpurity = ioModel.ReadGreenDat("greenUp.dat");
        }
        else if (spin == FermionSpin_t::Down)
        {
            greenImpurity = ioModel.ReadGreenDat("greenDown.dat");
        }

        using SelfCon_t = SelfCon::SelfConsistency<IOModel_t, Model_t, H0_t>;
        return std::make_unique<SelfCon_t>(SelfCon_t(jj, model, greenImpurity, spin));
    }

    return NULL;
} // namespace SelfCon

} // namespace SelfCon

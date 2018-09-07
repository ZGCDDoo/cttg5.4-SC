#pragma once

#include "SelfConsistency_DCA.hpp"

namespace SelfCon
{

std::unique_ptr<ABC_SelfConsistency> SelfConsistencyBuilder(const Json &jj, const FermionSpin_t &spin)
{
    const std::string modelType = jj["modelType"].get<std::string>();

    if (modelType == "SIAM_Square")
    {
        const size_t Nx = 1;
        using Model_t = Models::SIAM_Square;
        using IOModel_t = IO::IOSIAM;
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
    else if (modelType == "Square2x2_DCA")
    {
        const size_t Nx = 2;
        using Model_t = Models::ModelSquare2x2;
        using IOModel_t = IO::IOSquare2x2;
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
    else if (modelType == "Square4x4_DCA")
    {
        const size_t Nx = 4;
        using Model_t = Models::ModelSquare4x4_DCA;
        using IOModel_t = IO::IOSquare4x4_DCA;
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

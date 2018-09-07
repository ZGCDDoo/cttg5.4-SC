#pragma once

#include <string>
#include <vector>
#include <map>

namespace Conventions
{

using MapSS_t = std::map<std::string, std::string>;
using NameVector_t = std::vector<std::string>;

MapSS_t BuildFileNameConventions()
{
    MapSS_t nameCon;

    //Extensions
    const std::string datExt = ".dat";
    nameCon["datExt"] = datExt;

    const std::string jsonExt = ".json";
    nameCon["jsonExt"] = jsonExt;

    const std::string armaExt = ".arma";
    nameCon["armaExt"] = armaExt;

    //### Start Output files###
    nameCon["greenUpFile"] = "greenUp" + datExt;
    nameCon["greenDownFile"] = "greenDown" + datExt;

    nameCon["selfUpFile"] = "selfUp" + datExt;
    nameCon["selfDownFile"] = "selfDown" + datExt;

    nameCon["hybUpFile"] = "hybNextUp" + datExt;
    nameCon["hybDownFile"] = "hybNextDown" + datExt;

    nameCon["obsJsonFile"] = "Obs" + jsonExt;
    nameCon["updMeasJsonFile"] = "upd.meas" + jsonExt;
    nameCon["updThermJsonFile"] = "upd.therm" + jsonExt;

    nameCon["nUpMatrixFile"] = "nUpMatrix" + datExt;
    nameCon["nDownMatrixFile"] = "nDownMatrix" + datExt;

#ifdef DCA
    nameCon["tlocFile"] = "tloc_K" + armaExt;
    nameCon["hybFMFile"] = "hybFM_K" + armaExt;
#else
    nameCon["tlocFile"] = "tloc" + armaExt;
    nameCon["hybFMFile"] = "hybFM" + armaExt;
#endif

    nameCon["tktildeFile"] = "tktilde" + armaExt;

    nameCon["OutPutConventionFile"] = "outPutConvention" + datExt;
    //### End Output files###

    return nameCon;
}

NameVector_t BuildGreensVectorNames()
{
    NameVector_t nameVec;
    MapSS_t nameCon = BuildFileNameConventions();

    nameVec.push_back(nameCon["greenUpFile"]);
    nameVec.push_back(nameCon["greenDownFile"]);

    nameVec.push_back(nameCon["selfUpFile"]);
    nameVec.push_back(nameCon["selfDownFile"]);

    nameVec.push_back(nameCon["hybUpFile"]);
    nameVec.push_back(nameCon["hybDownFile"]);

    return nameVec;
}

} // namespace Conventions
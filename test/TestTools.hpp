#include <boost/filesystem.hpp>
#include <iostream>

namespace TestTools
{

using namespace boost::filesystem;

void RemoveFilesForTests()
{
    std::vector<std::string> files = {"tloc.arma", "tktilde.arma", "hybFM.arma", "config.dat"};

    for (const std::string &ss : files)
    {
        path filepath(ss);
        if (exists(filepath) && is_regular_file(filepath))
        {
            remove(filepath);
        }
    }
}

} // namespace TestTools
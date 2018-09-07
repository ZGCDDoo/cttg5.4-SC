#include <Version.hpp>
#include <iostream>

namespace PrintVersion
{
void PrintVersion()
{

    std::string gitBranch = GIT_BRANCH;
    std::string gitHash = GIT_COMMIT_HASH;

    std::cout << "\n\n\n";
    std::cout << "\t\t\t\t ============= CTTG5.3 =============" << std::endl;
    std::cout << "\t\t\t\t -----------------------------------" << std::endl;
    std::cout << "\t\t\t\t -----------------------------------" << std::endl;
    std::cout << "\t\t\t\t\t\t .... \n \n ";
    std::cout << "\t\t\t\t\t Git Branch = " << gitBranch << std::endl;
    std::cout << "\t\t\t\t\t gitHash = " << gitHash << std::endl;
    std::cout << "\t\t\t\t\t\t .... \n \n";
    std::cout << "\t\t\t\t -----------------------------------" << std::endl;
    std::cout << "\t\t\t\t -----------------------------------" << std::endl;
    std::cout << "\t\t\t\t ===================================" << std::endl;
    std::cout << "\n\n\n";
}

} // namespace PrintVersion
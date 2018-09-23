#Inspired by https://github.com/dmonopoly/gtest-cmake-example
#           https://github.com/misaka-10032/boost-mpi-examples

cmake_minimum_required(VERSION 2.8.2)

option(TEST "Build all tests." ON) # Makes boolean 'test' available.
option(HOME "home build" ON) #builds for linux, default
option(MAC "mac build" OFF)
option(GRAHAM "graham build" OFF)
option(MP2 "mp2 build" OFF)
option(MPI_BUILD "build with mpi" OFF)


if (${MP2})
    SET(CMAKE_CXX_COMPILER g++)
elseif (${MPI_BUILD})
    SET(CMAKE_CXX_COMPILER mpic++)
else ()
    SET(CMAKE_CXX_COMPILER clang++)
endif()


set(PROJECT_NAME cttg5.2)
project("${PROJECT_NAME}")


# Git version, etc, taken from: http://xit0.org/2013/04/cmake-use-git-branch-and-commit-details-in-project/
#   or https://github.com/pmirshad/cmake-with-git-metadata
#------------------------------------------------------------------------------------------------------------
# Get the current working branch
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the latest abbreviated commit hash of the working branch
execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)



message(STATUS "Git current branch: ${GIT_BRANCH}")
message(STATUS "Git commit hash: ${GIT_COMMIT_HASH}")

message(STATUS "Generating version.h")

configure_file(
  ${CMAKE_SOURCE_DIR}/src/Includes/Version.hpp.in
  ${CMAKE_BINARY_DIR}/generated/Version.hpp
)

include_directories(${CMAKE_BINARY_DIR}/generated)



#-------------------------------------------------------------------------------------------------------------


set(CMAKE_CXX_FLAGS " -Wall -Wextra -pedantic -std=c++1z -pipe -Ofast -march=native -fomit-frame-pointer") 
if(${MPI_BUILD})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVEMPI")
    set(TEST OFF)
endif()
if(${MPI_BUILD} AND ${HOME})
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    set(TEST OFF)
endif()

set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/src/Includes)
add_subdirectory(${PROJECT_SOURCE_DIR}/deps/cubature)

find_package(LAPACK REQUIRED)

#-------------HOME--------------------------------------------------
if (${HOME})
    find_package(Boost REQUIRED mpi serialization filesystem system)
    set(LIBRARIES_EXEC ${LAPACK_LIBRARIES} ${Boost_LIBRARIES} armadillo cubature)
endif ()
#-------------HOME--------------------------------------------------

if(${MAC})
    find_package(Boost REQUIRED mpi serialization filesystem system)
    set(LIBRARIES_EXEC lapack blas ${Boost_LIBRARIES} armadillo cubature)
endif()
 


#-------------GRAHAM--------------------------------------------------
if (${GRAHAM})
    set(TEST OFF)
    set(CMAKE_EXE_LINKER_FLAGS "-L${MKLROOT}/lib/intel64")
    set(LIBRARIES_EXEC armadillo mkl_gf_lp64 mkl_sequential mkl_core boost_filesystem boost_mpi cubature)
endif ()
#-------------GRAHAM--------------------------------------------------



#-------------MP2--------------------------------------------------
if (${MP2})
    set(TEST OFF)
    SET(CMAKE_EXE_LINKER_FLAGS "-L${MKLROOT}/lib/intel64 -L/opt/boost64/1.65.1/lib -L/opt/armadillo/8.300.0/usr/lib64")
    include_directories(/opt/armadillo/8.300.0/usr/include /opt/boost64/1.65.1/include/boost/  ${MKLROOT}/include)
    set(LIBRARIES_EXEC armadillo mkl_gf_lp64 mkl_sequential mkl_core boost_filesystem boost_mpi cubature)
endif ()
#-------------MP2--------------------------------------------------


################################

# Executables

################################
set(EXECECUTABLES
    cttg_sc
    #cttg_sub
    #cttg_afm
    #cttg_sub_afm
    #cttg_DCA
    #cttg_sub_DCA
)

foreach(executable ${EXECECUTABLES})
    list(APPEND ${executable}_src ${PROJECT_SOURCE_DIR}/src/${executable}.cpp)
    add_executable(${executable}
                ${${executable}_src}
                  )
endforeach()

foreach(executable ${EXECECUTABLES})
    target_link_libraries(${executable}  ${LIBRARIES_EXEC})
    install(TARGETS ${executable} DESTINATION $ENV{HOME}/bin)
endforeach()


#--------------------end Executables ----------------------------------------------------------


################################

# Testing

################################

if (${TEST})
    enable_testing()
    
    # Download and unpack googletest at configure time
    configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
    if(result)
        message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
    if(result)
        message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif()

    # Prevent overriding the parent project's compiler/linker
    # settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    # Add googletest directly to our build. This defines
    # the gtest and gtest_main targets.
    add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
                    ${CMAKE_BINARY_DIR}/googletest-build
                    EXCLUDE_FROM_ALL)

    # The gtest/gtest_main targets carry header search path
    # dependencies automatically when using CMake 2.8.11 or
    # later. Otherwise we have to add them here ourselves.
    if (CMAKE_VERSION VERSION_LESS 2.8.11)
        include_directories("${gtest_SOURCE_DIR}/include")
    endif()

    set(TESTS
    Fourier_DCATests
    FourierTests
    FillingAndDoccTests
    GreenBinningTests
    GreenMatTests
    GreenTauTests
    H0TriangleTests
    #HybFMAndTLocTests
    IntegratorTests
    ModelTriangle2x2Tests
    UtilitiesTests
    #SelfConsistencyTests
    MarkovChainTests
    MarkovChainSquare2x2Tests
    #MarkovChainAuxTests
    #MarkovChainSubMatrixSquare2x2Tests
    #MarkovChainAuxSubMatrixTests
    #MarkovChainSubMatrixTests
    MatrixTests
    ObservablesTests
    )

    set(TESTS_DIR ${PROJECT_SOURCE_DIR}/test)

    foreach(test ${TESTS})
        list(APPEND ${test}_src ${TESTS_DIR}/${test}.cpp)
        add_executable(${test}
                    ${${test}_src}
                    )
    endforeach()

    set(LIBRARIES_TEST ${LIBRARIES_EXEC} gtest gtest_main)
    foreach(test ${TESTS})
        target_link_libraries(${test} ${LIBRARIES_TEST})
    endforeach()

    foreach(test ${TESTS})
        add_test(NAME ${test} COMMAND ./${test})
    endforeach()

    #add_test(NAME ObservablesDMFTTests COMMAND "python3" ../test/obs_dmft_tests.py)
    #add_test(NAME ObservablesCDMFTTests COMMAND "python3" ../test/obs_cdmft_tests.py)
    #add_test(NAME ObservablesCDMFTBinningTests COMMAND "python3" ../test/obs_cdmft_binning_tests.py)
endif()


#--------------------end Testing ----------------------------------------------------------


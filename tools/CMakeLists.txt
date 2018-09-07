#Inspired by https://github.com/dmonopoly/gtest-cmake-example
#           https://github.com/misaka-10032/boost-mpi-examples

cmake_minimum_required(VERSION 2.8.2)

option(HOME "home build" ON) #builds for linux, default
option(MAC "mac build" OFF)
option(GRAHAM "graham build" OFF)
option(MP2 "mp2 build" OFF)

SET(CMAKE_CXX_COMPILER g++)


set(PROJECT_NAME Tools)
project("${PROJECT_NAME}")



set(CMAKE_CXX_FLAGS " -Wall -Wextra -pedantic -std=c++1z -pipe -Ofast -march=native -fomit-frame-pointer") 
#get_filename_component(PARENT_DIR ${Tools_DIR} DIRECTORY)

include_directories("${PROJECT_BINARY_DIR}")

find_package(LAPACK REQUIRED)

#-------------HOME--------------------------------------------------
if (${HOME})
    find_package(Boost REQUIRED mpi serialization filesystem system)
    set(LIBRARIES_EXEC ${LAPACK_LIBRARIES} ${Boost_LIBRARIES} armadillo)
endif ()
#-------------HOME--------------------------------------------------

if(${MAC})
    find_package(Boost REQUIRED mpi serialization filesystem system)
    set(LIBRARIES_EXEC lapack blas ${Boost_LIBRARIES} armadillo)
endif()
 


#-------------GRAHAM--------------------------------------------------
if (${GRAHAM})
    set(CMAKE_EXE_LINKER_FLAGS "-L${MKLROOT}/lib/intel64")
    set(LIBRARIES_EXEC armadillo mkl_gf_lp64 mkl_sequential mkl_core boost_filesystem boost_mpi)
endif ()
#-------------GRAHAM--------------------------------------------------



#-------------MP2--------------------------------------------------
if (${MP2})
    SET(CMAKE_EXE_LINKER_FLAGS "-L${MKLROOT}/lib/intel64 -L/opt/boost64/1.65.1/lib -L/opt/armadillo/8.300.0/usr/lib64")
    include_directories(/opt/armadillo/8.300.0/usr/include /opt/boost64/1.65.1/include/boost/  ${MKLROOT}/include)
    set(LIBRARIES_EXEC armadillo mkl_gf_lp64 mkl_sequential mkl_core boost_filesystem boost_mpi)
endif ()
#-------------MP2--------------------------------------------------


################################

# Executables

################################
set(EXECECUTABLES
    2x2K_To_R
)

foreach(executable ${EXECECUTABLES})
    list(APPEND ${executable}_src ${PROJECT_SOURCE_DIR}/${executable}.cpp)
    add_executable(${executable}
                ${${executable}_src}
                  )
endforeach()

foreach(executable ${EXECECUTABLES})
    target_link_libraries(${executable}  ${LIBRARIES_EXEC})
    install(TARGETS ${executable} DESTINATION $ENV{HOME}/bin)
endforeach()


#--------------------end Executables ----------------------------------------------------------




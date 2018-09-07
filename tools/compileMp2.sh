#!/bin/bash
module reset
module load cmake/3.6.1  gcc/6.1.0  intel64/17.4  boost64/1.65.1_intel17 openmpi/1.8.4_intel17  armadillo/8.300.0

g++ -o 2x2K_TO_R 2x2K_To_R.cpp \
-L${MKLROOT}/lib/intel64 -L/opt/boost64/1.65.1/lib -L/opt/armadillo/8.300.0/usr/lib64 \
-I/opt/armadillo/8.300.0/usr/include -I/opt/boost64/1.65.1/include/boost/  -I${MKLROOT}/include \
-larmadillo -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lboost_filesystem -std=c++14   

cp -i 2x2K_TO_R ~/bin/
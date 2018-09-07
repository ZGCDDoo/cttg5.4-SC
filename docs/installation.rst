.. _installation:

Installation
================================


**Note :**
If build problems,
please remove the build directory if it exists, then retry :
    
    $ rm -rf build

Dependencies
--------------
1. Armadillo
2. boost (mpi, serialization, filesystem, system)


Pre-Steps
----------
1. Make sure you have a "bin" directory in your home folder
2. Append the bin folder to your path. Add the following line to your ~/.bashrc:  export PATH="$PATH:~/bin"
3. $ source ~/.bashrc

Linux (Ubuntu 16.04)
----------------------
This installation procedure should work for many recent Linux flavors. For the following
we present the instructions specific for Ubuntu or derivatives.

1. Install the Dependencies
    $ sudo apt-get install libarmadillo-dev libboost-all-dev cmake
2. | $ mkdir build && cd build && cmake -DTEST=OFF .. && make -j NUMBER_OF_CORES install
   | # replace NUMBER_OF_CORE by say = 4


Mac
-----
This has been tested once. MPI not yet supported.

1. Install the Dependencies
    $ brew install armadillo boost
2. | $ mkdir build && cd build && cmake -DHOME=OFF -DMAC=ON  .. && make -j NUMBER_OF_CORES install
   | # replace NUMBER_OF_CORE by say = 4

Mp2
-----
1. $ module reset
2. $ module load cmake/3.6.1  gcc/6.1.0  intel64/17.4  boost64/1.65.1_intel17 openmpi/1.8.4_intel17  armadillo/8.300.0
3. $ mkdir build && cd build && cmake -DHOME=OFF -DMP2=ON -DMPI_BUILD=ON .. && make install


Graham and Ceder
-----------------
1. $ module reset 
2. $ module load nixpkgs/16.09  gcc/5.4.0 armadillo boost-mpi
3. | $ mkdir build && cd build && \\
   | cmake -DHOME=OFF -DGRAHAM=ON -DMPI_BUILD=ON  .. && make install
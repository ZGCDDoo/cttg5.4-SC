.. _tutorial:
.. include:: <isotech.txt>

Tutorial
=========

For Mp2 and Graham, use the "scractch" directories. If build problems,
please remove the build directory if it exists, then retry :
    
    $ rm -rf build


Graham: Tutorial 1
-------------------
1. | Connect to Graham:
   | $ ssh -X "user"@graham.computecanada.ca # where "user" is your compute canada/Mp2 Username
2. Ensure you have done the Pre-Steps described in :ref:`installation`.
3. $ salloc  --time=01:00:00 --ntasks=1 --mem-per-cpu=4000
4. Follow th Graham Procedure in :ref:`installation`.
5. $ module reset 
6. $ module load nixpkgs/16.09  gcc/5.4.0 armadillo boost-mpi
7. $ cd ../examples/CDMFT
8. $ cdmft_square4x4 params 1

This is  only an example, as the results will be wrong, because the 
Measurement time and the Updates bewteen measurements is to low.
To get sensible results, set MEASUREMENT_TIME to say 10, and UPDATES_MEAS
to 100.
 

Mp2: Tutorial 1
-------------------

1. Connect to Mp2:
2. Ensure you have done the Pre-Steps described in :ref:`installation`.
3. Follow th Mp2 Procedure in :ref:`installation`.
4. $ cd /path/to/cttg3.0/examples/CDMFT
5. $ cp ../Mp2/scriptMp2.pbs ./
6. Change the walltime to 01:00:00 and the queue: "qwork" -> "qtest"
7. Change "myExe=dmft" to "myExe=cdmft_square4x4"
8. qsub scriptMp2.pbs

The simulation should launch fairly quickly (in the hour).

This is  only an example, as the results will be wrong, because the 
Measurement time and the Updates bewteen measurements is to low.
To get sensible results, set MEASUREMENT_TIME to say 10, and UPDATES_MEAS
to 100.


Home
---------


Home: Tutorial 1 = dmft
^^^^^^^^^^^^^^^^^^^^^^^^
To Come. For now, go into the "examples folder", select your installation and run the bash script.
If there are some problems, then install "dos2unix" and run it on the bash files.

    $dos2unix runCDMFT

Ex:
    If you installed on your computer
    1. $ cd examples/Home
    2. $ bash runCDMFT.sh # in fact runs dmft


Home: tutorial 2 = cdmft_square4x4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. $ cd examples/CDMFT 
2. copy the script file, for example if on Home:
    $ cp ../Home/runCDMFT ./
3. in "runCDMFT", replace the line:
    myExe=dmft  -> myExe=cdmft_square4x4
4. $ bash runCDMFT


Launching a simulation
^^^^^^^^^^^^^^^^^^^^^^^^^
To launch a simulation, you need three files:

1. a script file, dependant on the platform (home, mp2, graham-cedar)
2. a "params" file (Ex: params1.json)
3. a "hyb" file (Ex: hyb1Up.dat)




When to use which algorithm
---------------------------

CT-Aux and CT-INT behave in a similar manner, and from my experience, one is not much faster than the other.

Cocerning Submatrix Updates, the codes "..._sub" should be used when the expansion order is high, say k>400
For k~<200, the algorithm may be slower.

Ex:
    cdmft_square4x4 for k < 400 
    cdmft_square4x4_sub for k > 400




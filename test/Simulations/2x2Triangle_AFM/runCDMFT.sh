

#!/bin/bash
iter=1
iterMax=100
myExe=cttg_afm
nprocess=4

rm tktilde.arma tloc.arma hybFM.arma config.dat
while [  $iter -lt $iterMax ];
    do
        mpirun -np $nprocess $myExe params $iter
	    iter=$[iter+1]
    done



#!/bin/bash
#PBS -N "test"
#PBS -l nodes=1
#PBS -l walltime=5:00:00
#PBS -q qwork@mp2
#PBS -j oe
cd ${PBS_O_WORKDIR}

module reset
module load cmake/3.6.1  gcc/6.1.0  intel64/17.4  boost64/1.65.1_intel17 openmpi/1.8.4_intel17  armadillo/8.300.0

export OMP_NUM_THREADS=1
nppn=24
nhosts=`wc -l < $PBS_NODEFILE`
let "NSLOTS = $nhosts*$nppn"

ITER=1
ITERMAX=10
myExe=cttg

if [ -a logfile ]
  then rm logfile
fi

rm tktilde.arma tloc.arma hybFM.arma config*.dat

while [ $ITER -le $ITERMAX ]
do
  echo begin iteration $ITER at: `date` >> logfile 
  
  mpirun -np $NSLOTS -machinefile $PBS_NODEFILE $myExe params ${ITER}
  
  echo end iteration $ITER at: `date` >> logfile
  ITER=$[$ITER+1]
done


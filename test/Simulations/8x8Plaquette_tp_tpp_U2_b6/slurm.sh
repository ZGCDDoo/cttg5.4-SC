#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=2000

## SBATCH --nodes=1
## SBATCH --mem=128000M
## SBATCH --ntasks-per-node=32


module reset
module load nixpkgs/16.09  gcc/5.4.0 armadillo boost-mpi

ITER=1
ITERMAX=10
myExe=cttg

if [ -a logfile ]
  then rm logfile
fi
rm config*.dat

while [ $ITER -le $ITERMAX ]
do
  echo begin iteration $ITER at: `date` >> logfile 

  srun $myExe params ${ITER}  

  echo end iteration $ITER at: `date` >> logfile
  ITER=$[$ITER+1]
done


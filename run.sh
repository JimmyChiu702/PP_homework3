#!/bin/bash 

length=10000
iteration=10000
seed=0

if [ $# -eq 3 ]
  then
    length=$1
    iteration=$2
    seed=$3
fi

/home/PP-f19/MPI/bin/mpiexec -npernode 1 --hostfile hostfile conduction $length $iteration $seed

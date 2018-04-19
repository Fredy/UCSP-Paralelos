#!/bin/bash

myRun() {
  echo "$PROCCESS process: 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072 "
  mpirun -np $PROCCESS ./mpi_odd_even.o  g 1024
  mpirun -np $PROCCESS ./mpi_odd_even.o  g 2048
  mpirun -np $PROCCESS ./mpi_odd_even.o  g 4096
  mpirun -np $PROCCESS ./mpi_odd_even.o  g 8192
  mpirun -np $PROCCESS ./mpi_odd_even.o  g 16384
  mpirun -np $PROCCESS ./mpi_odd_even.o  g 32768
  mpirun -np $PROCCESS ./mpi_odd_even.o  g 65536
}
    
PROCCESS=1
myRun

PROCCESS=2
myRun

PROCCESS=4
myRun

PROCCESS=8
myRun

PROCCESS=16
myRun


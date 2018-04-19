#!/bin/bash

myRun() {
  echo "$PROCCESS process: 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072 "
  mpirun -np $PROCCESS ./mpi_mat_vect_time.o 1024 1024
  mpirun -np $PROCCESS ./mpi_mat_vect_time.o 2048 2048
  mpirun -np $PROCCESS ./mpi_mat_vect_time.o 4096 4096
  mpirun -np $PROCCESS ./mpi_mat_vect_time.o 8192 8192
  mpirun -np $PROCCESS ./mpi_mat_vect_time.o 16384 16384
  # mpirun -np $PROCCESS ./mpi_mat_vect_time.o 32768 32768
  # mpirun -np $PROCCESS ./mpi_mat_vect_time.o 65536 65536
  # mpirun -np $PROCCESS ./mpi_mat_vect_time.o 131072 131072
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


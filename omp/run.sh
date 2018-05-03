#!/bin/bash


mRun() {
  echo "$PROCESS"
  ./mat_vec.out $PROCESS 8000000 8
  ./mat_vec.out $PROCESS 80000 800
  ./mat_vec.out $PROCESS 8000 8000
  ./mat_vec.out $PROCESS 800 80000
  ./mat_vec.out $PROCESS 8 8000000
}

PROCESS=1
mRun

PROCESS=2
mRun

PROCESS=4
mRun
PROCESS=8
mRun

#!/bin/bash

mRun() {
  echo "$PROCESS"
  ./odd_even1.out $PROCESS 10000 g
  ./odd_even2.out $PROCESS 10000 g
}

PROCESS=1
mRun

PROCESS=2
mRun

PROCESS=4
mRun
PROCESS=8
mRun

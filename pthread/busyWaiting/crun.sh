#!/bin/bash

echo "1024, 10240, 102400, 102400, 10240000, 102400000, 1024000000" 

mRun() {
  echo "$PROC p." 
  ./cbusy.out $PROC 1024
  ./cbusy.out $PROC 10240
  ./cbusy.out $PROC 102400
  ./cbusy.out $PROC 1024000
  ./cbusy.out $PROC 10240000
  ./cbusy.out $PROC 102400000
  ./cbusy.out $PROC 1024000000
}

PROC=1;
mRun

PROC=2;
mRun

PROC=4;
mRun

PROC=8;
mRun

PROC=16;
mRun

PROC=32;
mRun

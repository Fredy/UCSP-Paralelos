#!/bin/bash

echo "1024, 10240, 102400, 102400, 10240000, 102400000, 1024000000" 

mRun() {
  echo "$PROC p." 
  ./cmutex.out $PROC 1024
  ./cmutex.out $PROC 10240
  ./cmutex.out $PROC 102400
  ./cmutex.out $PROC 1024000
  ./cmutex.out $PROC 10240000
  ./cmutex.out $PROC 102400000
  ./cmutex.out $PROC 1024000000
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

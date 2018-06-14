#!/bin/sh

mkdir -p build
cd build

#./../../cmake3_11_3/bin/cmake .. -DCMAKE_BUILD_TYPE=Release  \
#						    -DCMAKE_CXX_FLAGS="-O3 -std=c++14 -I/home/faalvarez/tbb2018/include -L/home/faalvarez/tbb2018/lib/intel64/gcc4.7/" \
#						    -DCMAKE_CXX_COMPILER=/opt/shared/gcc_5_4_0/bin/g++ -DCMAKE_C_COMPILER=/opt/shared/gcc_5_4_0/bin/gcc 
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -std=c++14"

make

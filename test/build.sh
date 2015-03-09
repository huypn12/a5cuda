#!/bin/sh

cd ../ && make
cd test
cp ../A5Cuda.so ./
#g++ -std=c++11 -fPIC -Wall -c main.cpp
#g++ -ldl -L/usr/local/cuda/lib64 -lcuda -lcudart -L/usr/lib64 -lboost_system -lboost_thread -o main main.o ../A5Cuda.so
g++ -O3 -w -o main main.cpp ../A5CudaStubs.cpp  -Wl,--no-as-needed -ldl -L/usr/local/cuda/lib64 -lcuda -lcudart -lboost_system -lboost_thread 


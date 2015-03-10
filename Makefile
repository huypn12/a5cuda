# Makefile for A5Cuda
# Build to A5Cuda.so
# Deploying on development environment, Ubuntu 1404, Cuda 6.5, GF104 Quadro 1000M


NVCC            = /usr/local/cuda/bin/nvcc
NVCCFLAGS       = -O3 -gencode arch=compute_20,code=sm_21 -Xcompiler -fpic
CC              = g++ 
CFLAGS          = -Wall -fPIC
INCLUDE         = -I/usr/local/cuda/include
LD_LIBRARY_PATH = -fPIC -lstdc++ -ldl -L/usr/local/cuda/lib64 -lcuda -lcudart -L/usr/lib64 -lboost_system -lboost_thread
LINKER          = gcc -shared 

all: A5Cuda.o A5CudaKernel.o A5CudaSlice.o Advance.o
	$(LINKER) $(LD_LIBRARY_PATH) -o A5Cuda.so A5CudaKernel.o A5Cuda.o A5CudaSlice.o Advance.o

A5CudaKernel.o: kernel/A5CudaKernel.cu kernel/A5CudaKernel.h
	$(NVCC) $(NVCCFLAGS) -c kernel/A5CudaKernel.cu

A5CudaSlice.o: A5CudaSlice.cu A5CudaSlice.h
	$(NVCC) $(NVCCFLAGS) -c A5CudaSlice.cu

A5Cuda.o: A5Cuda.cpp A5Cuda.h
	$(CC) $(CFLAGS) $(INCLUDE) -c A5Cuda.cpp

Advance.o: Advance.cpp Advance.h
	$(CC) $(CFLAGS) -c Advance.cpp

clean:
	rm -rf *.o

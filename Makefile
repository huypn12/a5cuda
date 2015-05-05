# Makefile for A5Cuda
# Build to A5Cuda.so
# Deploying on development environment, Ubuntu 1404, Cuda 6.5, GF104 Quadro 1000M


NVCC      = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -O3 -std=c++11 -Xcompiler -fPIC --compile --relocatable-device-code=false --default-stream per-thread
GENCODE   = -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30

LINKER  = $(NVCC)
LDFLAGS = --cudart static --shared --relocatable-device-code=false -link

LIB_OBJS = kernel/A5CudaKernel.o A5CudaSlice.o A5Cuda.o Advance.o

all: A5Cuda.so

A5Cuda.so:$(LIB_OBJS)
	$(LINKER) $(LDFLAGS) $(GENCODE) -o A5Cuda.so $(LIB_OBJS)

kernel/A5CudaKernel.o:kernel/A5CudaKernel.cu 
	$(NVCC) $(NVCCFLAGS) $(GENCODE) -x cu -o $@ -c $<

A5CudaSlice.o:A5CudaSlice.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE) -x cu -o $@ -c $<

A5Cuda.o:A5Cuda.cpp 
	$(NVCC) $(NVCCFLAGS) $(GENCODE) -x c++ -o $@ -c $<

Advance.o:Advance.cpp 
	$(NVCC) $(NVCCFLAGS) $(GENCODE) -x c++ -o $@ -c $<

clean:
	rm -rf *.o *.so test/*.o test/*.so

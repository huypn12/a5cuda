#ifndef __A5_CUDA_KERNEL__
#define __A5_CUDA_KERNEL__

#include <cuda_runtime.h>
#include <stdint.h>

extern "C"
__global__ void a51_cuda_kernel(
        unsigned int rounds,
        unsigned int size,
        unsigned int dps,
        uint4* stopVals,
        unsigned int* finished
);

__global__ void a51_cuda_kernel_1(
        unsigned int rounds,
        unsigned int size,
        unsigned int dps,
        uint4* stopVals,
        unsigned int* finished
);

#endif

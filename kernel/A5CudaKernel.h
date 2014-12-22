#ifndef __A5_CUDA_KERNEL__
#define __A5_CUDA_KERNEL__

#include <cuda_runtime.h>
#include <stdint.h>

extern "C"
__global__ void a5_cuda_kernel(
        unsigned int rounds,
        unsigned int size,
        unsigned int condition,
        uint64_t* stopVals,
        uint64_t* roundfuncs,
        unsigned int* control
);

#endif

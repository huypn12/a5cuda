#ifndef __A5_CUDA_KERNEL__
#define __A5_CUDA_KERNEL__

#include <cuda_runtime.h>
#include <vector_types.h>
#include <stdint.h>

#define N_LFSR_CONSTANTS 9
#define N_RUNNING_PARAM_CONSTANTS 4

__host__ cudaError_t copy_kernel_constants(
        unsigned int* source
        );

__global__ void a51_cuda_kernel(
        uint4* states,
        unsigned int* control
);

#endif

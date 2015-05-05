#ifndef __A5_CUDA_KERNEL__
#define __A5_CUDA_KERNEL__

#include <cuda_runtime.h>
#include <stdint.h>

extern "C"

#define N_LFSR_CONSTANTS 9

__constant__ unsigned int LFSR_CONSTANTS[N_LFSR_CONSTANTS] = {
    0x000100, 0x000400, 0x000400,   // CLK
    0x072000, 0x300000, 0x700080,   // TAPS
    0x040000, 0x200000, 0x400000    // OUT
};

#define R1CLK LFSR_CONSTANTS[1]
#define R2CLK LFSR_CONSTANTS[2]
#define R3CLK LFSR_CONSTANTS[3]

#define R1TAPS LFSR_CONSTANTS[4]
#define R2TAPS LFSR_CONSTANTS[5]
#define R3TAPS LFSR_CONSTANTS[6]

#define R1OUT LFSR_CONSTANTS[7]
#define R2OUT LFSR_CONSTANTS[8]
#define R3OUT LFSR_CONSTANTS[9]

__constant__ unsigned int RUNNING_PARAM_CONSTANTS[4];
//{ROUNDS, DATA_SIZE, DPS}
__global__ void a51_cuda_kernel(
        /*
        unsigned int rounds,
        unsigned int size,
        unsigned int dps,
        */
        uint4* stopVals,
        unsigned int* finished
);

#endif

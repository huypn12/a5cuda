#include "A5CudaKernel.h"

#define N_LFSR_CONSTANTS 9
__device__ __constant__ unsigned int LFSR_CONSTANTS[ N_LFSR_CONSTANTS ] = {
    0x000100, 0x000400, 0x000400,   // CLK
    0x072000, 0x300000, 0x700080,   // TAPS
    0x040000, 0x200000, 0x400000    // OUT
};

#define R1CLK   LFSR_CONSTANTS[0]
#define R2CLK   LFSR_CONSTANTS[1]
#define R3CLK   LFSR_CONSTANTS[2]

#define R1TAPS  LFSR_CONSTANTS[3]
#define R2TAPS  LFSR_CONSTANTS[4]
#define R3TAPS  LFSR_CONSTANTS[5]

#define R1OUT   LFSR_CONSTANTS[6]
#define R2OUT   LFSR_CONSTANTS[7]
#define R3OUT   LFSR_CONSTANTS[8]

#define N_RUNNING_PARAM_CONSTANTS 4
__device__ __constant__ unsigned int RUNNING_PARAM_CONSTANTS[ N_RUNNING_PARAM_CONSTANTS ];

#define M_ITERCOUNT     RUNNING_PARAM_CONSTANTS[0]
#define M_DPS           RUNNING_PARAM_CONSTANTS[1]
#define M_DATASIZE      RUNNING_PARAM_CONSTANTS[2]

__host__ cudaError_t copy_kernel_constants(unsigned int* src)
{
    return ( cudaMemcpyToSymbol(RUNNING_PARAM_CONSTANTS, src, N_RUNNING_PARAM_CONSTANTS*sizeof(unsigned int)) );
}

// Calculate parity
__device__  int parity32(unsigned int x)
{
    x ^= x >> 16;
    x ^= x >> 8;
    x ^= x >> 4;
    x ^= x >> 2;
    x ^= x >> 1;
    return (x&1);
}

// Calculate parity for a byte
__device__ int parity8(unsigned char x) {
    x ^= x>>4;
    x ^= x>>2;
    x ^= x>>1;
    return x&1;
}

// Calculate parity for 2 bits
__device__ int parity2(unsigned char x) {
    x ^= x>>1;
    return x&1;
}

__device__ int clock_lfsr1(unsigned int reg)
{
    unsigned char t = (reg & R1TAPS) >> 13; // get only bits from 18-13
    reg = (reg << 1);
    reg |= parity8(t);

    return reg;
}

__device__ int clock_lfsr2(unsigned int reg)
{
    unsigned char taps = (reg & R2TAPS) >> 20; // get only 2 bits from 20 -> 21
    reg = (reg << 1);
    reg |= parity2(taps);
    return reg;
}

__device__ int clock_lfsr3(unsigned int reg)
{
    unsigned int taps = reg & R3TAPS; //  Tung: no need for t
    reg = (reg << 1);
    reg |= parity32(taps);
    return reg;
}

__device__ int majority(
        unsigned int R1,
        unsigned int R2,
        unsigned int R3
        )
{
    // Tung: Use shift -> faster than parity
    int sum = ((R1&R1CLK)>>8) + ((R2&R2CLK)>>10) + ((R3&R3CLK)>>10);
    return sum >> 1;
}

// Clock majority
__device__  void clock_major (unsigned int& R1, unsigned int& R2, unsigned int &R3)
{
    // new clock using bitshift
    int major = majority(R1, R2, R3);
    int majr1 = (((R1 & R1CLK) != 0) == major);
    R1 = (((majr1 << 31) >> 31) & clock_lfsr1(R1)) | (((!majr1 << 31) >> 31) & R1);
    majr1 = (((R2 & R2CLK) != 0) == major);
    R2 = (((majr1 << 31) >> 31) & clock_lfsr2(R2)) | (((!majr1 << 31) >> 31) & R2);
    majr1 = (((R3 & R3CLK) != 0) == major);
    R3 = (((majr1 << 31) >> 31) & clock_lfsr3(R3)) | (((!majr1 << 31) >> 31) & R3);
}

/**
 * kernel, process each of 32 chains slot
 */
__global__ void a51_cuda_kernel (
        /*
           unsigned int rounds,
           unsigned int size,
           unsigned int dp,
         */
        uint4* states,
        unsigned int* controls
        )
{
    // huyphung: bitsliced, using 4x32 bit integer instead of 2x64 bit word

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (M_DATASIZE)) {
        return;
    }

    uint4 state = states[idx];
    unsigned int control = controls[idx];

    unsigned int res_lo = state.x;
    unsigned int res_hi = state.y;
    unsigned int key_lo = state.z;
    unsigned int key_hi = state.w;
    unsigned int last_key_hi = 0;
    unsigned int last_key_lo = 0;

    unsigned int R1;
    unsigned int R2;
    unsigned int R3;

    unsigned int tval;

    if ((res_hi | res_lo) == 0) {
        return;
    }

    for (register int r = 0; r < M_ITERCOUNT; r++) {
        last_key_hi = res_hi;
        last_key_lo = res_lo;
        res_hi ^= key_hi;
        res_lo ^= key_lo;

        /* //This statement cause slowdown by broking memory coalescing
           if (((res_hi >> dp) | control[idx]) == 0) {
           states[idx].x = last_key_lo;
           states[idx].y = last_key_hi;
           return;
           }
         */
        if (((res_hi >> M_DPS) | control) == 0) {
            res_hi = last_key_hi;
            res_lo = last_key_lo;
            break;
        }

        R1 = res_lo & 0x7FFFF;
        R2 = ((res_hi & 0x1FF) << 13) | (res_lo >> 19);
        R3 = res_hi >> 9;

        // Discarded clocking
        for (register int i = 0; i < 99; ) {
            clock_major(R1, R2, R3);
            clock_major(R1, R2, R3);
            clock_major(R1, R2, R3);
            i=i+3;
        }

        // TODO: reduce reversebit step in kernel by change output method
        // saved clocking
        tval = 0;

        res_hi = 0ULL;
        for (register int i = 0; i < 32; i++) {
            clock_major(R1, R2, R3);
            tval = ((R1>>18)^(R2>>21)^(R3>>22)) & 0x1;
            res_hi = res_hi | (tval << (31-i));
        }

        res_lo = 0ULL;
        for (register int i = 0; i < 32; i++) {
            clock_major(R1, R2, R3);
            tval = ((R1>>18)^(R2>>21)^(R3>>22)) & 0x1;
            res_lo = res_lo | (tval << (31-i));
        }

    }
    states[idx].x = res_lo;
    states[idx].y = res_hi;
    __syncthreads();
}


#include "A5CudaKernel.h"

/*@huypn: using constant memory here drastically decrease performance

#define R1CLK   LFSR_CONSTANTS[0]
#define R2CLK   LFSR_CONSTANTS[1]
#define R3CLK   LFSR_CONSTANTS[2]

#define R1TAPS  LFSR_CONSTANTS[3]
#define R2TAPS  LFSR_CONSTANTS[4]
#define R3TAPS  LFSR_CONSTANTS[5]

#define R1OUT   LFSR_CONSTANTS[6]
#define R2OUT   LFSR_CONSTANTS[7]
#define R3OUT   LFSR_CONSTANTS[8]

*/


#define R1CLK   c_lfsr1_clk
#define R2CLK   c_lfsr2_clk 
#define R3CLK   c_lfsr3_clk

const unsigned int c_lfsr1_clk = 0x00010;
const unsigned int c_lfsr2_clk = 0x00040;
const unsigned int c_lfsr3_clk = 0x00040;

#define R1TAPS  c_lfsr1_taps
#define R2TAPS  c_lfsr2_taps
#define R3TAPS  c_lfsr3_taps

const unsigned int c_lfsr1_taps = 0x072000;
const unsigned int c_lfsr2_taps = 0x300000;
const unsigned int c_lfsr3_taps = 0x700080;

/*@huypn there is no need of feedback mask, since only one bit is count

#define R1OUT   c_lfsr1_out
#define R2OUT   c_lfsr2_out
#define R3OUT   c_lfsr3_out

const unsigned int c_lfsr1_out = 0x040000;
const unsigned int c_lfsr2_out = 0x200000;
const unsigned int c_lfsr3_out = 0x400000;

*/


#define N_RUNNING_PARAM_CONSTANTS 4
__device__ __constant__ unsigned int RUNNING_PARAM_CONSTANTS[ N_RUNNING_PARAM_CONSTANTS ];

#define ITERCOUNT     RUNNING_PARAM_CONSTANTS[0]
#define DPS           RUNNING_PARAM_CONSTANTS[1]
#define DATASIZE      RUNNING_PARAM_CONSTANTS[2]
#define WARPSIZE      RUNNING_PARAM_CONSTANTS[3]


__host__ cudaError_t copy_kernel_constants(unsigned int* src)
{
    return ( cudaMemcpyToSymbol(RUNNING_PARAM_CONSTANTS, src, N_RUNNING_PARAM_CONSTANTS*sizeof(unsigned int)) );
}

/*@huypn remove inline parity function to decrease uses of registers 

// Calculate parity
__device__  __inline__ int parity32(unsigned int x)
{
    x ^= x >> 16;
    x ^= x >> 8;
    x ^= x >> 4;
    x ^= x >> 2;
    x ^= x >> 1;
    return (x&1);
}

// Calculate parity for a byte
__device__ __inline__ int parity8(unsigned char x) {
    x ^= x>>4;
    x ^= x>>2;
    x ^= x>>1;
    return x&1;
}u

// Calculate parity for 2 bits
__device__ __inline__ int parity2(unsigned char x) {
    x ^= x>>1;
    return x&1;
}

*/

__device__ __inline__ int clock_lfsr1(unsigned int reg)
{
    unsigned char taps;
    taps = (reg & R1TAPS) >> 13; // get only bits from 18-13
    taps ^= taps >> 4;
    taps ^= taps >> 2;
    taps ^= taps >> 1;
    taps &= 1;
    reg <<= 1;
    reg |= taps;

    return reg;
}

__device__ __inline__ int clock_lfsr2(unsigned int reg)
{
    unsigned char taps;
    taps = (reg & R2TAPS) >> 20; // get only 2 bits from 20 -> 21
    taps ^= taps >> 1;
    taps &= 1;
    reg <<= 1;
    reg |= taps;

    return reg;
}

__device__ __inline__ int clock_lfsr3(unsigned int reg)
{
    unsigned int taps;
    taps = reg & R3TAPS; //  Tung: no need for t
    taps ^= taps >> 16;
    taps ^= taps >> 8;
    taps ^= taps >> 4;
    taps ^= taps >> 2;
    taps ^= taps >> 1;
    taps &= 1;
    reg <<= 1;
    reg |= taps;

    return reg;
}

/*@huypn remove majority function to decrease uses of registers

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

*/

__device__  void clock_major (unsigned int& R1, unsigned int& R2, unsigned int &R3)
{
    // new clock using bitshift
    int major = ((R1&R1CLK)>>8) + ((R2&R2CLK)>>10) + ((R3&R3CLK)>>10); //majority(R1, R2, R3);
    major >>= 1;
    major = (((R1 & R1CLK) != 0) == major);
    R1 = (((major << 31) >> 31) & clock_lfsr1(R1)) | (((!major << 31) >> 31) & R1);
    major = (((R2 & R2CLK) != 0) == major);
    R2 = (((major << 31) >> 31) & clock_lfsr2(R2)) | (((!major << 31) >> 31) & R2);
    major = (((R3 & R3CLK) != 0) == major);
    R3 = (((major << 31) >> 31) & clock_lfsr3(R3)) | (((!major << 31) >> 31) & R3);
}

/**
 * kernel, process each of 32 chains slot
 */
__global__ void a51_cuda_kernel (
        /* @huypn Using builtin supported types uint4 is better than using unsigned int array
           unsigned int rounds,
           unsigned int size,
           unsigned int dp,
         */
        uint4* states,
        unsigned int* controls
        )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (DATASIZE)) {
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

    for (int r = 0; r < ITERCOUNT; r++) {
        last_key_hi = res_hi;
        last_key_lo = res_lo;
        res_hi ^= key_hi;
        res_lo ^= key_lo;

        /*@huypn This statement cause slowdown by breaking memory coalescing

           if (((res_hi >> dp) | control[idx]) == 0) {
           states[idx].x = last_key_lo;
           states[idx].y = last_key_hi;
           return;
           }

         */
        if (((res_hi >> DPS) | control) == 0) {
            res_hi = last_key_hi;
            res_lo = last_key_lo;
            break;
        }

        R1 = res_lo & 0x7FFFF;
        R2 = ((res_hi & 0x1FF) << 13) | (res_lo >> 19);
        R3 = res_hi >> 9;

        // Discarded clocking
        int major = 0;
        for (int i = 0; i < 99; i++) {
            /*@huypn remove loop unrolling

            clock_major(R1, R2, R3);
            clock_major(R1, R2, R3);
            clock_major(R1, R2, R3);
            i=i+3;

            */

            major = (((R1&R1CLK)>>8) + ((R2&R2CLK)>>10) + ((R3&R3CLK)>>10)) >> 1; /*major >>= 1;*/ //majority(R1, R2, R3);
            major = (((R1 & R1CLK) != 0) == major);
            R1 = (((major << 31) >> 31) & clock_lfsr1(R1)) | (((!major << 31) >> 31) & R1);
            major = (((R2 & R2CLK) != 0) == major);
            R2 = (((major << 31) >> 31) & clock_lfsr2(R2)) | (((!major << 31) >> 31) & R2);
            major = (((R3 & R3CLK) != 0) == major);
            R3 = (((major << 31) >> 31) & clock_lfsr3(R3)) | (((!major << 31) >> 31) & R3);

        }

        /*TODO: reduce reversebit step in kernel
         *      by change output method saved clocking
         */
        tval = 0;

        res_hi = 0;
        for (int i = 0; i < 32; i++) {
            /*clock_major(R1, R2, R3);*/
            major = (((R1&R1CLK)>>8) + ((R2&R2CLK)>>10) + ((R3&R3CLK)>>10)) >> 1; /*major >>= 1;*/ //majority(R1, R2, R3);
            major = (((R1 & R1CLK) != 0) == major);
            R1 = (((major << 31) >> 31) & clock_lfsr1(R1)) | (((!major << 31) >> 31) & R1);
            major = (((R2 & R2CLK) != 0) == major);
            R2 = (((major << 31) >> 31) & clock_lfsr2(R2)) | (((!major << 31) >> 31) & R2);
            major = (((R3 & R3CLK) != 0) == major);
            R3 = (((major << 31) >> 31) & clock_lfsr3(R3)) | (((!major << 31) >> 31) & R3);

            tval = ((R1>>18)^(R2>>21)^(R3>>22)) & 0x1;
            res_hi = res_hi | (tval << (31-i));
        }

        res_lo = 0;
        for (int i = 0; i < 32; i++) {
            /*clock_major(R1, R2, R3);*/
            major = (((R1&R1CLK)>>8) + ((R2&R2CLK)>>10) + ((R3&R3CLK)>>10)) >> 1; /*major >>= 1;*/ //majority(R1, R2, R3);
            major = (((R1 & R1CLK) != 0) == major);
            R1 = (((major << 31) >> 31) & clock_lfsr1(R1)) | (((!major << 31) >> 31) & R1);
            major = (((R2 & R2CLK) != 0) == major);
            R2 = (((major << 31) >> 31) & clock_lfsr2(R2)) | (((!major << 31) >> 31) & R2);
            major = (((R3 & R3CLK) != 0) == major);
            R3 = (((major << 31) >> 31) & clock_lfsr3(R3)) | (((!major << 31) >> 31) & R3);

            tval = ((R1>>18)^(R2>>21)^(R3>>22)) & 0x1;
            res_lo = res_lo | (tval << (31-i));
        }

    }
    states[idx].x = res_lo;
    states[idx].y = res_hi;
}


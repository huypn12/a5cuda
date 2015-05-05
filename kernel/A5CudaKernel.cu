#include "A5CudaKernel.h"

/*
// Clock bit$
#define R1CLK   0x000100
#define R2CLK   0x000400
#define R3CLK   0x000400

// Feedback tapping bits
#define R1TAPS  0x072000
#define R2TAPS  0x300000
#define R3TAPS  0x700080

// huyphung: seeking for an alternative for this mask
// Output tapping bits
#define R1OUT   0x040000
#define R2OUT   0x200000
#define R3OUT   0x400000
*/

// Calculate parity
__device__  int parity32(uint64_t x)
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
        unsigned int rounds,
        unsigned int size,
        unsigned int dp,
        uint4* states,
        unsigned int* controls
        )
{
    // huyphung: bitsliced, using 4x32 bit integer instead of 2x64 bit word

    int idx;
    idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    unsigned int res_lo = states[idx].x;
    unsigned int res_hi = states[idx].y;
    unsigned int key_lo = states[idx].z;
    unsigned int key_hi = states[idx].w;
    unsigned int control = controls[idx];
    unsigned int last_key_hi = 0;
    unsigned int last_key_lo = 0;

    unsigned int R1;
    unsigned int R2;
    unsigned int R3;

    if ((res_hi | res_lo) == 0) {
        return;
    }

    for (register int r = 0; r < rounds; r++) {
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
        if (((res_hi >> dp) | control) == 0) {
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
        unsigned int tval = 0;

        res_hi = 0ULL;
        for (int i = 0; i < 32; ) {
            clock_major(R1, R2, R3);
            tval = ((R1>>18)^(R2>>21)^(R3>>22)) & 0x1;
            res_hi = res_hi | (tval << (31-i));
            i += 2;
        }

        res_lo = 0ULL;
        for (int i = 0; i < 32; ) {
            clock_major(R1, R2, R3);
            tval = ((R1>>18)^(R2>>21)^(R3>>22)) & 0x1;
            res_lo = res_lo | (tval << (31-i));
            i += 2;
        }

    }
    states[idx].x = res_lo;
    states[idx].y = res_hi;
}

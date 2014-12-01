#include "A5CudaKernel.h"

// Clock bit$
#define R1CLK   0x000100
#define R2CLK   0x000400
#define R3CLK   0x000400

// Feedback tapping bits
#define R1TAPS  0x072000
#define R2TAPS  0x300000
#define R3TAPS  0x700080

// Output tapping bits
#define R1OUT   0x040000
#define R2OUT   0x200000
#define R3OUT   0x400000

#define BLOCK_SIZE 16

// Calculate parity
__device__ inline int parity(uint64_t x)
{
    x ^= x >> 16;
    x ^= x >> 8;
    x ^= x >> 4;
    x ^= x >> 2;
    x ^= x >> 1;
    return (x&1);
}

// Calculate parity for a byte
__device__ inline
int parityByte(unsigned char x) {
    x ^= x>>4;
    x ^= x>>2;
    x ^= x>>1;
    return x&1;
}

// Calculate parity for 2 bits
__device__ inline
int parity2Bits(unsigned char x) {
    x ^= x>>1;
    return x&1;
}

// Clock one register
__device__ inline
int32_t clockOne (uint32_t reg, uint32_t taps)
{
    taps = reg & taps; //  Tung: no need for t
    reg = (reg << 1);
    reg |= parity(taps);
    return reg;
}

__device__ inline
int32_t clockOne_R1(uint32_t reg)
{
    unsigned char t = (reg & R1TAPS) >> 13; // get only bits from 18-13
    reg = (reg << 1);
    reg |= parityByte(t);
    return reg;
}

__device__ inline
int32_t clockOne_R2(uint32_t reg)
{
    unsigned char taps = (reg & R2TAPS) >> 20; // get only 2 bits from 20 -> 21
    reg = (reg << 1);
    reg |= parity2Bits(taps);
    return reg;
}

// Calculate majority
__device__ inline int32_t majority (uint32_t R1, uint32_t R2, uint32_t R3)
{
    // Tung: Use shift -> faster than parity
    int32_t sum = ((R1&R1CLK)>>8) + ((R2&R2CLK)>>10) + ((R3&R3CLK)>>10);
    return sum >> 1;
}

// Clock majority
__device__ inline void clockMajor (uint32_t& R1, uint32_t& R2, uint32_t &R3)
{
    // new clock using bitshift
    int32_t major = majority(R1, R2, R3);
    int32_t majr1 = (((R1 & R1CLK) != 0) == major);
    R1 = (((majr1 << 31) >> 31) & clockOne_R1(R1)) | (((!majr1 << 31) >> 31) & R1);
    majr1 = (((R2 & R2CLK) != 0) == major);
    R2 = (((majr1 << 31) >> 31) & clockOne_R2(R2)) | (((!majr1 << 31) >> 31) & R2);
    majr1 = (((R3 & R3CLK) != 0) == major);
    R3 = (((majr1 << 31) >> 31) & clockOne(R3, R3TAPS)) | (((!majr1 << 31) >> 31) & R3);
}

// Reverse bits
__device__ inline
uint64_t reversebits (uint64_t R, int length)
{
    R = ((R >> 1) & 0x5555555555555555) | ((R & 0x5555555555555555) << 1);
    // swap consecutive pairs
    R = ((R >> 2) & 0x3333333333333333) | ((R & 0x3333333333333333) << 2);
    // swap nibbles ...
    R = ((R >> 4) & 0x0F0F0F0F0F0F0F0F) | ((R & 0x0F0F0F0F0F0F0F0F) << 4);
    // swap bytes
    R = ((R >> 8) & 0x00FF00FF00FF00FF) | ((R & 0x00FF00FF00FF00FF) << 8);
    // swap 2-byte long pairs
    R = ((R >> 16) & 0x0000FFFF0000FFFF) | ((R & 0x0000FFFF0000FFFF)  << 16);
    // last
    R = ( (R >> 32) | (R << 32));

    R = R >> (64-length);
    return R;
}

__device__ inline
int64_t selectbitwise(int64_t a, int64_t b, int64_t test)
{
    test = !!test;
    test = ((test << 63) >> 63) & a;
    int64_t t2 = !test;
    test |= ((t2 << 63) >> 63) & b;
    return test;
}

/**
 * kernel, process each of 32 chain slots
 */
__global__ void a5_cuda_kernel (
        unsigned int rounds,
        unsigned int size,
        uint64_t* state,
        uint64_t* roundfuncs,
        unsigned int* finished
        )
{
    int idx;
    idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    if (finished[idx]) {
        return;
    }

     __shared__ unsigned int ok; ok = finished[idx];
     __shared__ uint64_t res; res = state[idx];
     __shared__ uint64_t rf; rf = roundfuncs[idx];

    __shared__ uint32_t R1;
    __shared__ uint32_t R2;
    __shared__ uint32_t R3;

    for (register int r = 0; r < rounds; r++) {
        // Put output to register
        // Tung: reversbits only one
        res = reversebits(res, 64);
        R1 = res & 0x7FFFF;
        R2 = (res & 0x1FFFFF80000) >> 19;
        R3 = (res & 0xFFFFFE0000000000) >> 41;

        // Discarded clocking
        for (register int i = 0; i < 99; ) {
            clockMajor(R1, R2, R3);
            clockMajor(R1, R2, R3);
            clockMajor(R1, R2, R3);
            i += 3;
        }

        // saved clocking
        res = 0ULL;
        for (register int i = 0; i < 64;) {
            clockMajor(R1, R2, R3);
            res |= ((0ULL | (((R1&R1OUT)>>18)^((R2&R2OUT)>>21)^((R3&R3OUT)>>22))) << i);
            ++i;
            clockMajor(R1, R2, R3);
            res |= ((0ULL | (((R1&R1OUT)>>18)^((R2&R2OUT)>>21)^((R3&R3OUT)>>22))) << i);
            ++i;
        }

        res ^= rf;

        if ((res & 0xFFF) == 0) {
            ok = 1;
            break;
        }
    }
    state[idx] = res;
    finished[idx] = ok;
}

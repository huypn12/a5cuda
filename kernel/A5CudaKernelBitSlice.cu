#include <stdio.h>
#include <stdint.h>

#include "Advance.h"



__device__ inline
int PopcountNibble(int x) {
    int res = 0;
    for (int i=0; i<4; i++) {
        res += x & 0x01;
        x = x >> 1;
    }
    return res;
}

__device__ inline
int calcClockMask(unsigned int clocks)
{
    int k = clocks & 0xf;
    int j = (clocks >> 4) & 0xf;
    int i = (clocks >> 8) & 0xf;

    /* Copy input */
    int m1 = i;
    int m2 = j;
    int m3 = k;
    /* Generate masks */
    int cm1 = 0;
    int cm2 = 0;
    int cm3 = 0;
    /* Counter R2 */
    int r2count = 0;
    for (int l = 0; l < 4 ; l++ ) {
        cm1 = cm1 << 1;
        cm2 = cm2 << 1;
        cm3 = cm3 << 1;
        int maj = ((m1>>3)+(m2>>3)+(m3>>3))>>1;
        if ((m1>>3)==maj) {
            m1 = (m1<<1)&0x0f;
            cm1 |= 0x01;
        }
        if ((m2>>3)==maj) {
            m2 = (m2<<1)&0x0f;
            cm2 |= 0x01;
            r2count++;
        }
        if ((m3>>3)==maj) {
            m3 = (m3<<1)&0x0f;
            cm3 |= 0x01;
        }
    }
    return ((r2count<<12) | (cm1<<8) | (cm2<<4) | cm3);
}

__device__ inline
unsigned int calcTable6bit(unsigned int lfsr)
{
    int j = lfsr & 0xf;
    int i = (lfsr >> 4) & 0x3f;

    int count = PopcountNibble(j);
    int feedback = 0;
    int data = i;
    for (int k=0; k<count; k++) {
        feedback = feedback << 1;
        int v = (data>>5) ^ (data>>4) ^ (data>>3);
        data = data << 1;
        feedback ^= (v&0x01);
    }
    data = i;
    int mask = j;
    int output = 0;
    for (int k=0; k<4; k++) {
        output = (output<<1) ^ ((data>>5)&0x01);
        if (mask&0x08) {
            data = data << 1;
        }
        mask = mask << 1;
    }
    return ((feedback<<4) | output);
}

__device__ inline
unsigned int calcTable5bit(unsigned int lfsr)
{
    int j = lfsr & 0xf;
    int i = (lfsr >> 4) & 0x1f;

    int count = PopcountNibble(j);
    int feedback = 0;
    int data = i;
    for (int k=0; k<count; k++) {
        feedback = feedback << 1;
        int v = (data>>4) ^ (data>>3);
        data = data << 1;
        feedback ^= (v&0x01);
    }
    data = i;
    int mask = j;
    int output = 0;
    for (int k=0; k<4; k++) {
        output = (output<<1) ^ ((data>>4)&0x01);
        if (mask&0x08) {
            data = data << 1;
        }
        mask = mask << 1;
    }
    return ((feedback<<4) | output);

}

__device__ inline
unsigned int calcTable4bit(unsigned int lfsr)
{
    int j = lfsr & 0xf;
    int i = (lfsr >> 4) & 0xf;

    int count = PopcountNibble(j);
    int feedback = 0;
    int data = i;
    for (int k=0; k<count; k++) {
        feedback = feedback << 1;
        int v = (data>>3);
        data = data << 1;
        feedback ^= (v&0x01);
    }
    return ((count<<4)|feedback);

}

__global__
a5_cuda_kernel_bitslice(
        unsigned int cycles,
        unsigned int size,
        unsigned int condition,
        uint64_t* states,
        uint64_t* advances,
        unsigned int* finished)
{

    int threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    if (finished[idx]) {
        return;
    }

    unsigned int out_hi = states[i] >> 32;
    unsigned int out_lo = state[i];
    
    out_lo = out_lo ^ advances[threadIdx];
    out_hi = out_hi ^ advances[threadIdx] >> 32;

    unsigned int last_key_lo;
    unsigned int last_key_hi;

    bool keysearch = (target != 0ULL);

//    for (int round=start_round; round < stop_round; ) {
    for (int cycle = 0; cycle < cycles; cycle++) {

        if ((out_hi>>mCondition)==0) {
            // uint64_t res = (((uint64_t)out_hi)<<32)|out_lo;
            // res = ReverseBits(res);
            // printf("New round %i %016llx %08x:%08x\n", round, res, out_hi, out_lo);
            round++;
            if (round>=stop_round) break;
        }

        unsigned int lfsr1 = out_lo;
        unsigned int lfsr2 = (out_hi << 13) | (out_lo >> 19);
        unsigned int lfsr3 = out_hi >> 9;

        last_key_hi = out_hi;
        last_key_lo = out_lo;

        for (int i=0; i<25 ; i++) {
            int clocks = ((lfsr1<<3)&0xf00) | ((lfsr2>>3)&0xf0) | ((lfsr3>>7)&0xf);
            int masks = calcClockMask(clocks);

            /* lfsr1 */
            unsigned int tmask = (masks>>8)&0x0f;
            unsigned int tval = calcTable6bit(((lfsr1>>9)&0x3f0)|tmask);
            unsigned int tval2 = calcTable4bit(((lfsr1>>6)&0xf0)|tmask);
            lfsr1 = (lfsr1<<(tval2>>4))^(tval>>4)^(tval2&0x0f);

            /* lfsr2 */
            tmask = (masks>>4)&0x0f;
            tval = calcTable5bit(((lfsr2>>13)&0x1f0)|tmask);
            out_hi = out_hi ^ (tval&0x0f);
            lfsr2 = (lfsr2<<(masks>>12))^(tval>>4);

            /* lfsr3 */
            tmask = masks & 0x0f;
            tval = calcTable6bit(((lfsr3>>13)&0x3f0)|tmask);
            tval2 = calcTable4bit((lfsr3&0xf0)|tmask);
            lfsr3 = (lfsr3<<(tval2>>4))^(tval>>4)^(tval2&0x0f);
        }
        for (int i=0; i<8 ; i++) {
            int clocks = ((lfsr1<<3)&0xf00) | ((lfsr2>>3)&0xf0) | ((lfsr3>>7)&0xf);
            int masks = calcClockMask(clocks);

            /* lfsr1 */
            unsigned int tmask = (masks>>8)&0x0f;
            unsigned int tval = calcTable6bit(((lfsr1>>9)&0x3f0)|tmask);
            unsigned int tval2 = calcTable4bit(((lfsr1>>6)&0xf0)|tmask);

            out_hi = (out_hi << 4) | (tval&0x0f);
            lfsr1 = (lfsr1<<(tval2>>4))^(tval>>4)^(tval2&0x0f);

            /* lfsr2 */
            tmask = (masks>>4)&0x0f;
            tval = calcTable5bit(((lfsr2>>13)&0x1f0)|tmask);
            out_hi = out_hi ^ (tval&0x0f);
            lfsr2 = (lfsr2<<(masks>>12))^(tval>>4);

            /* lfsr3 */
            tmask = masks & 0x0f;
            tval = calcTable6bit(((lfsr3>>13)&0x3f0)|tmask);
            out_hi =  out_hi ^ (tval&0x0f);
            tval2 = calcTable4bit((lfsr3&0xf0)|tmask);
            lfsr3 = (lfsr3<<(tval2>>4))^(tval>>4)^(tval2&0x0f);
        }
        for (int i=0; i<8 ; i++) {
            int clocks = ((lfsr1<<3)&0xf00) | ((lfsr2>>3)&0xf0) | ((lfsr3>>7)&0xf);
            int masks = calcClockMask(clocks);

            /* lfsr1 */
            unsigned int tmask = (masks>>8)&0x0f;
            unsigned int tval = calcTable6bit(((lfsr1>>9)&0x3f0)|tmask);
            out_lo = (out_lo << 4) | (tval&0x0f);
            unsigned int tval2 = calcTable4bit(((lfsr1>>6)&0xf0)|tmask);
            lfsr1 = (lfsr1<<(tval2>>4))^(tval>>4)^(tval2&0x0f);

            /* lfsr2 */
            tmask = (masks>>4)&0x0f;
            tval = calcTable5bit(((lfsr2>>13)&0x1f0)|tmask);
            out_lo = out_lo ^ (tval&0x0f);
            lfsr2 = (lfsr2<<(masks>>12))^(tval>>4);

            /* lfsr3 */
            tmask = masks & 0x0f;
            tval = calcTable6bit(((lfsr3>>13)&0x3f0)|tmask);
            out_lo =  out_lo ^ (tval&0x0f);
            tval2 = calcTable4bit((lfsr3&0xf0)|tmask);
            lfsr3 = (lfsr3<<(tval2>>4))^(tval>>4)^(tval2&0x0f);
        }
        if (keysearch&&(target_hi==out_hi)&&(target_lo==out_lo)) {
            /* report key as finishing state */
            out_hi = last_key_hi;
            out_lo = last_key_lo;
            //start_round = -1;
            finished[idx] = 1;
            break;
        }
    }

    /* Report completed chains */
    /*
    uint64_t res = (((uint64_t)out_hi)<<32)|out_lo;
    res = ReverseBits(res);
    return res;
    */
    state[i] = (((uint64_t) out_hi) << 32) | out_lo;

}


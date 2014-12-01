#include "A5CudaKernel.h"


    __global__
void a5_cuda_kernel_bitslice()
{
    out_lo = out_lo ^ RFtable[2*round];
    out_hi = out_hi ^ RFtable[2*round+1];

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
        int masks = mClockMask[clocks];

        /* lfsr1 */
        unsigned int tmask = (masks>>8)&0x0f;
        unsigned int tval = mTable6bit[((lfsr1>>9)&0x3f0)|tmask];
        unsigned int tval2 = mTable4bit[((lfsr1>>6)&0xf0)|tmask];
        lfsr1 = (lfsr1<<(tval2>>4))^(tval>>4)^(tval2&0x0f);

        /* lfsr2 */
        tmask = (masks>>4)&0x0f;
        tval = mTable5bit[((lfsr2>>13)&0x1f0)|tmask];
        out_hi = out_hi ^ (tval&0x0f);
        lfsr2 = (lfsr2<<(masks>>12))^(tval>>4);

        /* lfsr3 */
        tmask = masks & 0x0f;
        tval = mTable6bit[((lfsr3>>13)&0x3f0)|tmask];
        tval2 = mTable4bit[(lfsr3&0xf0)|tmask];
        lfsr3 = (lfsr3<<(tval2>>4))^(tval>>4)^(tval2&0x0f);
    }
    for (int i=0; i<8 ; i++) {
        int clocks = ((lfsr1<<3)&0xf00) | ((lfsr2>>3)&0xf0) | ((lfsr3>>7)&0xf);
        int masks = mClockMask[clocks];

        /* lfsr1 */
        unsigned int tmask = (masks>>8)&0x0f;
        unsigned int tval = mTable6bit[((lfsr1>>9)&0x3f0)|tmask];
        out_hi = (out_hi << 4) | (tval&0x0f);
        unsigned int tval2 = mTable4bit[((lfsr1>>6)&0xf0)|tmask];
        lfsr1 = (lfsr1<<(tval2>>4))^(tval>>4)^(tval2&0x0f);

        /* lfsr2 */
        tmask = (masks>>4)&0x0f;
        tval = mTable5bit[((lfsr2>>13)&0x1f0)|tmask];
        out_hi = out_hi ^ (tval&0x0f);
        lfsr2 = (lfsr2<<(masks>>12))^(tval>>4);

        /* lfsr3 */
        tmask = masks & 0x0f;
        tval = mTable6bit[((lfsr3>>13)&0x3f0)|tmask];
        out_hi =  out_hi ^ (tval&0x0f);
        tval2 = mTable4bit[(lfsr3&0xf0)|tmask];
        lfsr3 = (lfsr3<<(tval2>>4))^(tval>>4)^(tval2&0x0f);
    }
    for (int i=0; i<8 ; i++) {
        int clocks = ((lfsr1<<3)&0xf00) | ((lfsr2>>3)&0xf0) | ((lfsr3>>7)&0xf);
        int masks = mClockMask[clocks];

        /* lfsr1 */
        unsigned int tmask = (masks>>8)&0x0f;
        unsigned int tval = mTable6bit[((lfsr1>>9)&0x3f0)|tmask];
        out_lo = (out_lo << 4) | (tval&0x0f);
        unsigned int tval2 = mTable4bit[((lfsr1>>6)&0xf0)|tmask];
        lfsr1 = (lfsr1<<(tval2>>4))^(tval>>4)^(tval2&0x0f);

        /* lfsr2 */
        tmask = (masks>>4)&0x0f;
        tval = mTable5bit[((lfsr2>>13)&0x1f0)|tmask];
        out_lo = out_lo ^ (tval&0x0f);
        lfsr2 = (lfsr2<<(masks>>12))^(tval>>4);

        /* lfsr3 */
        tmask = masks & 0x0f;
        tval = mTable6bit[((lfsr3>>13)&0x3f0)|tmask];
        out_lo =  out_lo ^ (tval&0x0f);
        tval2 = mTable4bit[(lfsr3&0xf0)|tmask];
        lfsr3 = (lfsr3<<(tval2>>4))^(tval>>4)^(tval2&0x0f);
    }
    if (keysearch&&(target_hi==out_hi)&&(target_lo==out_lo)) {
        /* report key as finishing state */
        out_hi = last_key_hi;
        out_lo = last_key_lo;
        start_round = -1;
        break;
    }
}


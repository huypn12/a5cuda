#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include "../A5CudaStubs.h"

void test0()
{
    uint64_t plain = 0;
    uint64_t start_val;
    uint64_t stop_val;
    int* dummy = 0;
    /* Count samples that may be checked from known plaintext */
    int samples = 0;
    const char* plaintext =
        "101011110011100101110000001010010101110111010001110111101111100011";
    const char* ch = plaintext;
    while (*ch == '0' || *ch == '1') {
        ch++;
        samples++;
    }
    samples -= 63;
    int submitted = 0;
    for (int count = 0; count < 9; count ++) {
        for (int i = 0; i < samples; i++) {
            uint64_t plain = 0;
            uint64_t plainrev = 0;
            for (int j = 0; j < 64; j++) {
                if (plaintext[i + j] == '1') {
                    plain = plain | (1ULL << j);
                    plainrev = plainrev | (1ULL << (63 - j));
                }
            }
            for (int k = 0; k < 8; k++) {
                A5CudaSubmit(plain, k, 140, dummy);
            }
        }
    }
}

void push_burst_2000()
{
    int *dummy;
    uint64_t feed = 0xfeacdfea13456760;
    uint64_t plain = 0;
    for (int i=0; i < 125; i++) {
        for (int j=0; j < 8; j++) {
           plain = feed + i*j;
           A5CudaSubmit(plain, j, 140, (void**)dummy);
        }
    }
    for (int i=0; i < 125; i++) {
        for (int j=0; j < 8; j++) {
           plain = feed + i*j;
           A5CudaSubmitPartial(plain, j, 140, (void**)dummy);
        }
    }
}

void pop_result(int n)
{
    uint64_t start_val;
    uint64_t stop_val;
    int* dummy;
    for (int i=0; i < n; i++) {
        while (not A5CudaPopResult(start_val, stop_val, (void**)dummy)) {
            usleep(100);
        }
    }
}

int main(int argc, char* argv[]) {
    A5CudaInit(8,12);
    push_burst_2000();
    pop_result(2000);
    A5CudaShutdown();
}


#include <stdio.h>
#include <sys/time.h>

#include "../A5CudaStubs.h"

int main(int argc, char* argv[]) {
    A5CudaInit(8,12);
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
    struct timeval start_time;
    gettimeofday(&start_time, NULL);
    for (int i = 0; i < samples; i++) {
        uint64_t plain = 0;
        uint64_t plainrev = 0;
        for (int j = 0; j < 64; j++) {
            if (plaintext[i + j] == '1') {
                plain = plain | (1ULL << j);
                plainrev = plainrev | (1ULL << (63 - j));
            }
        }
        printf("%lx\n", plain);
        for (int k = 0; k < 8; k++) {
            for (int c = 0; c < 999999; c++)
                A5CudaSubmit(plain, k, 140, dummy);
        }
    }
    while (not A5CudaPopResult(start_val, stop_val, (void**) &dummy)) {
    }
    struct timeval stop_time;
    gettimeofday(&stop_time, NULL);
    unsigned long diff = 1000000 * (stop_time.tv_sec - start_time.tv_sec);
    diff += stop_time.tv_usec - start_time.tv_usec;
    printf("%i msec\t\n", (int) (diff / 1000));
    A5CudaShutdown();
}

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

#include <iostream>
#include <stdint.h>
#include <time.h>

#include "A51CudaKernel.h"

#define MEMORY_ALIGNMENT 4096
#define ALIGN_UP(x, size) (((size_t)x + (size -1)) & (~(size - 1)))

#define DATA_SIZE 4096
#define CHUNK_SIZE 32
#define BLOCK_SIZE 32


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

uint64_t AdvanceRFlfsr(uint64_t v)
{
    for (int i = 64; i > 0; --i) {
        uint64_t fb = (
                ~(
                    ((v >> 63) & 1) ^
                    ((v >> 62) & 1) ^
                    ((v >> 60) & 1) ^
                    ((v >> 59) & 1)
                 )
                ) & 1;
        v <<= 1;
        v |= fb;
    }
    return v;
}

uint64_t* getRounds(unsigned int id, unsigned int size)
{
    uint64_t* mAdvances = new uint64_t[size];

    uint64_t r = 0;
    for (int i = id; i > 0; --i) r = AdvanceRFlfsr(r);

    for (int i = 0; i<size; i++) {
        uint64_t r2 = reversebits(r, 64);
        mAdvances[i] = r;
        r = AdvanceRFlfsr(r);
    }

    return mAdvances;
}

bool simulate_poprequest(uint64_t &startValue, unsigned int &startRound, unsigned int &stopRound)
{
    startValue = 0x1f7b8bba940e9cf5;
    startRound = 2;
    stopRound = 5;
    return true;

}

bool simulate_pushresult(uint64_t startValue, uint64_t stopValue, uint64_t startRound, uint64_t stopRound)
{
    std::cout << std::hex << "(" << startRound << "," << stopRound << ") " << std::hex << startValue << "  " << std::hex << stopValue;
    return true;
}

void kernelTest_mapped()
{
    clock_t startClk = clock();
    std::cout << "Memory allocation" << std::endl;

    cudaEvent_t allocStart, allocStop;
    cudaEventCreate(&allocStart);
    cudaEventCreate(&allocStop);
    cudaEventRecord(allocStart, 0);

    uint64_t* h_UAstartValues = (uint64_t *) malloc (DATA_SIZE * sizeof(uint64_t) + MEMORY_ALIGNMENT);
    uint64_t* h_startValues = (uint64_t *) ALIGN_UP(h_UAstartValues, MEMORY_ALIGNMENT);

    uint64_t* h_UAstopValues = (uint64_t *) malloc (DATA_SIZE * sizeof(uint64_t) + MEMORY_ALIGNMENT);
    uint64_t* h_stopValues = (uint64_t *) ALIGN_UP(h_UAstopValues, MEMORY_ALIGNMENT);

    unsigned int* h_UAstartRounds = (unsigned int *) malloc (DATA_SIZE * sizeof(unsigned int) + MEMORY_ALIGNMENT);
    unsigned int* h_startRounds = (unsigned int *) ALIGN_UP(h_UAstartRounds, MEMORY_ALIGNMENT);

    unsigned int* h_UAstopRounds = (unsigned int *) malloc (DATA_SIZE * sizeof(unsigned int) + MEMORY_ALIGNMENT);
    unsigned int* h_stopRounds = (unsigned int *) ALIGN_UP(h_UAstopRounds, MEMORY_ALIGNMENT);

    unsigned int* h_UAcurrentRounds = (unsigned int *) malloc (DATA_SIZE * sizeof(unsigned int) + MEMORY_ALIGNMENT);
    unsigned int* h_currentRounds = (unsigned int *) ALIGN_UP(h_UAcurrentRounds, MEMORY_ALIGNMENT);

    uint64_t* h_UAstates = (uint64_t *) malloc (DATA_SIZE * sizeof(uint64_t) + MEMORY_ALIGNMENT);
    uint64_t* h_states = (uint64_t *) ALIGN_UP(h_UAstates, MEMORY_ALIGNMENT);
    checkCudaErrors(cudaHostRegister(h_states, DATA_SIZE * sizeof(uint64_t), CU_MEMHOSTALLOC_DEVICEMAP));

    uint64_t* h_UAroundfuncs = (uint64_t *) malloc (DATA_SIZE * sizeof(uint64_t) + MEMORY_ALIGNMENT);
    uint64_t* h_roundfuncs = (uint64_t *) ALIGN_UP(h_UAroundfuncs, MEMORY_ALIGNMENT);
    checkCudaErrors(cudaHostRegister(h_roundfuncs, DATA_SIZE * sizeof(uint64_t), CU_MEMHOSTALLOC_DEVICEMAP));

    unsigned int* h_UAfinished = (unsigned int *) malloc (DATA_SIZE * sizeof(unsigned int) + MEMORY_ALIGNMENT);
    unsigned int* h_finished = (unsigned int *) ALIGN_UP(h_UAfinished, MEMORY_ALIGNMENT);
    checkCudaErrors(cudaHostRegister(h_finished, DATA_SIZE * sizeof(unsigned int), CU_MEMHOSTALLOC_DEVICEMAP));


    // initiate memory
    std::cout << "mapping pointer" << std::endl;

    uint64_t* d_roundfuncs;
    uint64_t* d_states;
    unsigned int* d_finished;
    checkCudaErrors(cudaHostGetDevicePointer((void**)&d_roundfuncs, h_roundfuncs, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void**)&d_finished, h_finished, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void**)&d_states, h_states, 0));

    cudaEventRecord(allocStop, 0);
    cudaEventSynchronize(allocStop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, allocStart, allocStop);
    std::cout << elapsedTime << std::endl;


    // repeat kernel until all chain slot satisfies
    cudaEvent_t kernelStart, kernelStop;
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);
    cudaEventRecord(kernelStart, 0);

    unsigned int maxCycles = 3000000;
    unsigned int nCycles = 128;
    unsigned int cycleCount = 0;

    // populate value
    uint64_t valueSeed  = 0x1f7b8bba940e9cf5;
    uint64_t* roundfuncs = getRounds(140, 8);
    for (unsigned int i = 0; i < DATA_SIZE; i++) {
        h_startValues[i] = valueSeed;
        h_startRounds[i] = i % 8;
        h_stopRounds[i] = 7;
        h_currentRounds[i] = h_startRounds[i];
        h_roundfuncs[i] = roundfuncs[h_startRounds[i]];
        h_finished[i] = 0;
        h_states[i] = h_startValues[i] ^ h_roundfuncs[i];
    }

    // repeating kernel
    // if satisfies maxround of lastbits, switchround
    bool todo = true;
    while (todo) {
        for (unsigned int i = 0; i < DATA_SIZE; i++) {
            // completed single round, reached maxcycles or finish round state
            if (h_currentRounds[i] < h_stopRounds[i]) {
                if (h_finished[i] || (cycleCount >= maxCycles))
                {
                    h_finished[i] = 0;
                    h_currentRounds[i] += 1;
                    h_roundfuncs[i] = roundfuncs[h_currentRounds[i]];
                }
            }
            // completed final round, pushrequest, pop result 
            else if (h_finished[i] && (h_currentRounds[i] == h_stopRounds[i]))
            {
                simulate_pushresult(h_startValues[i], h_states[i], h_startRounds[i], h_stopRounds[i]);
                uint64_t newState;
                unsigned int newStartRound;
                unsigned int newStopRound;
                if (simulate_poprequest(newState, newStartRound, newStopRound))
                {
                    h_startValues[i] = newState;
                    h_startRounds[i] = newStartRound;
                    h_stopRounds[i] = newStopRound;
                    h_states[i] = newState ^ roundfuncs[newStartRound];
                    h_finished[i] = 0;
                }
            }
        }

        // do kernel
        std::cout << "invoking kernel" << std::endl;
        dim3 dimBlock(BLOCK_SIZE, 1);
        dim3 dimGrid(( DATA_SIZE - 1) / BLOCK_SIZE + 1, 1);
        a51_cuda_kernel<<<dimBlock, dimGrid>>>(nCycles, DATA_SIZE, d_states, d_roundfuncs, d_finished);
        cudaDeviceSynchronize();
        cycleCount += nCycles;

    }

    cudaEventRecord(kernelStop, 0);
    cudaEventSynchronize(kernelStop);
    elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, kernelStart, kernelStop);
    std::cout << elapsedTime << std::endl;

    clock_t stopClk = clock();
    float seconds = (float)(stopClk - startClk) / CLOCKS_PER_SEC;
    std::cout << std::endl << "......" << std::endl << seconds << std::endl;
}

int main()
{

    kernelTest_mapped();

}

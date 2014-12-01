
#include <cuda.h>
#include <cuda_runtime.h>

#include "utils/helper_cuda.h"
#include "A5CudaSlice.h"

/**
 * Author: Huy Phung
 * Constructor
 * @param: A5Cuda, device id, number of zero bits ending, maximum round
 * @return:
 */
A5CudaSlice::A5CudaSlice(
        A5Cuda* _controller,
        int deviceId,
        int dp,
        unsigned int _maxRound)
{
    // set cuda deviceid
    cudaSetDevice(deviceId);

    maxCycles = 3000000;
    nCycles = 256;
    state = ePopulate;
    maxRound = _maxRound;
    finishedMask = ~(0xffffffffffffffffULL << dp);

    // set controller
    controller = _controller;

    // allocate memory
    jobs = new A5Cuda::JobPiece_s[DATA_SIZE];

    // need allocating and aligning memory to pin on host
    uint64_t* h_UAstates = (uint64_t *) malloc (DATA_SIZE * sizeof(uint64_t) + MEMORY_ALIGNMENT);
    hm_states   = (uint64_t *) ALIGN_UP(h_UAstates, MEMORY_ALIGNMENT);
    checkCudaErrors(cudaHostRegister(hm_states, DATA_SIZE * sizeof(uint64_t), CU_MEMHOSTALLOC_DEVICEMAP));

    uint64_t* h_UAroundfuncs = (uint64_t *) malloc (DATA_SIZE * sizeof(uint64_t) + MEMORY_ALIGNMENT);
    hm_roundfuncs   = (uint64_t *) ALIGN_UP(h_UAroundfuncs, MEMORY_ALIGNMENT);
    checkCudaErrors(cudaHostRegister(hm_roundfuncs, DATA_SIZE * sizeof(uint64_t), CU_MEMHOSTALLOC_DEVICEMAP));

    unsigned int* h_UAfinished = (unsigned int *) malloc (DATA_SIZE * sizeof(unsigned int) + MEMORY_ALIGNMENT);
    hm_finished   = (unsigned int *) ALIGN_UP(h_UAfinished, MEMORY_ALIGNMENT);
    checkCudaErrors(cudaHostRegister(hm_finished, DATA_SIZE * sizeof(unsigned int), CU_MEMHOSTALLOC_DEVICEMAP));

    // map to device
    checkCudaErrors(cudaHostGetDevicePointer((void**)&d_roundfuncs, hm_roundfuncs, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void**)&d_finished, hm_finished, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void**)&d_states, hm_states, 0));

}

/**
 * Author: Huy Phung
 * Destructor
 * @param:
 * @return:
 */
A5CudaSlice::~A5CudaSlice()
{
    cudaHostUnregister(hm_states);
    cudaHostUnregister(hm_roundfuncs);
    cudaHostUnregister(hm_finished);
    delete [] jobs;
    cudaDeviceReset();
}

/**
 * Author: Huy Phung
 * tick: to change state of a slice and do corresponding task
 * @param:
 * @return:
 */
void A5CudaSlice::tick()
{
    switch (state)
    {
        case ePopulate:
            populate();
            state = eProcess;
            break;

        case eKernel:
            invokeKernel();
            state = eProcess;
            break;

        case eProcess:
            process();
            state = eKernel;
            break;

        default:
            // std::cout << "state error" << std::endl;
            break;

    }
}

/**
 * Author: Huy Phung
 * zeroing state of slots
 * @param:
 * @return:
 */
void A5CudaSlice::populate()
{
    for (unsigned int i = 0; i < DATA_SIZE; i++)
    {
        jobs[i].start_value   = 0;
        jobs[i].end_value     = 0;
        jobs[i].start_round   = 0;
        jobs[i].current_round = 0;
        jobs[i].end_round     = 0;
        jobs[i].round_func    = NULL;
        jobs[i].cycles        = 0;
        jobs[i].idle          = true;
        hm_finished[i]        = CHAINSTATE_FINISHED;
        hm_states[i]          = 0;
        hm_roundfuncs[i]      = 0;
    }
}

/**
 * Author: Huy Phung
 * process slots after GPU has done
 * @param:
 * @return:
 */
void A5CudaSlice::process()
{
    for (unsigned int i = 0; i < DATA_SIZE; i++) {
        A5Cuda::JobPiece_s* currentJob = &jobs[i];
        // idle item, populate with a new request
        if (currentJob->idle) {
            // populate new job
            if (controller->PopRequest(currentJob)) {
                unsigned int startRound = currentJob->start_round;
                hm_roundfuncs[i]        = currentJob->round_func[startRound];
                hm_states[i]            = currentJob->round_func[startRound] ^ currentJob->start_value;
                hm_finished[i]          = CHAINSTATE_RUNNING;
            } else {
            }
        } else {
            bool intMerge = currentJob->cycles > maxCycles;
            // int merge, if a chain went through the maximum cycles without reaching the finish state, just drop it
            if (intMerge) {
                hm_finished[i] = CHAINSTATE_FINISHED;
                currentJob->end_value = 0xffffffffffffffffULL;
                controller->PushResult(currentJob);
                currentJob->idle = true;
                if (controller->PopRequest(currentJob)) {
                    currentJob->idle = false;
                    hm_roundfuncs[i] = currentJob->round_func[currentJob->start_value];
                    hm_states[i]     = currentJob->start_value ^ hm_roundfuncs[i];
                    hm_finished[i]   = CHAINSTATE_RUNNING;
                } else {
                }
            } else {
                if (hm_finished[i] == CHAINSTATE_FINISHED) {
                    // reached the end round
                    if (currentJob->current_round == currentJob->end_round) {
                        // full search, rear submit
                        if (currentJob->end_round == maxRound - 1) {
                            currentJob->end_value = hm_states[i];
                            controller->PushResult(currentJob);
                            currentJob->idle = true;
                            if (controller->PopRequest(currentJob)) {
                                currentJob->idle = false;
                                assert (currentJob != NULL);
                                hm_roundfuncs[i] = currentJob->round_func[currentJob->start_round];
                                hm_states[i]     = currentJob->start_value ^ hm_roundfuncs[i];
                                hm_finished[i]   = CHAINSTATE_RUNNING;
                            }
                        // partial search, front submit
                        } else if (currentJob->end_round < maxRound - 1) {
                            currentJob->end_value = hm_states[i] ^ hm_roundfuncs[i];
                            controller->PushResult(currentJob);
                            if (controller->PopRequest(currentJob)) {
                                currentJob->idle = false;
                                hm_roundfuncs[i] = currentJob->round_func[currentJob->start_round];
                                hm_states[i]     = currentJob->start_value ^ hm_roundfuncs[i];
                                hm_finished[i]   = CHAINSTATE_RUNNING;
                            }
                        }
                    // in process, just continue
                    } else if (currentJob->current_round < currentJob->end_round) {
                        currentJob->current_round   += 1;
                        hm_roundfuncs[i]            = currentJob->round_func[currentJob->current_round];
                        hm_finished[i]              = CHAINSTATE_RUNNING;
                    }
                }
            }
        }
    }
}

/**
 * Author: Huy Phung
 * invoking cuda kernel
 * ver 1.0: synchronous kernel call
 * TODO: asynchronous kernel(s) call
 * @param:
 * @return:
 */
void A5CudaSlice::invokeKernel()
{
    dim3 dimBlock(BLOCK_SIZE, 1);
    dim3 dimGrid((DATA_SIZE - 1) / BLOCK_SIZE + 1, 1);
    a5_cuda_kernel<<<dimBlock, dimGrid>>>(nCycles, DATA_SIZE, d_states, d_roundfuncs, d_finished);
    cudaDeviceSynchronize();
    // increase number of cycle for each slot
    for(unsigned int i = 0; i < DATA_SIZE; i++) {
        jobs[i].cycles += nCycles;
    }
    return;
}


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
        A5Cuda* a5cuda,
        int deviceId,
        int mCondition,
        unsigned int max_round)
{
    // set cuda deviceid
    cudaSetDevice(deviceId);
    
    // set running parameter 
    mMaxCycles = 3000000;
    mCycles = 256;
    mMaxRound = mMaxRound;
    mDp = mCondition;

    // set controller
    controller = a5cuda;

    // allocate memory
    jobs = new A5Cuda::JobPiece_s[DATA_SIZE];

    // need allocating and aligning memory to pin on host
    uint64_t* h_UAstates = (uint64_t *) malloc (DATA_SIZE * sizeof(uint64_t) + MEMORY_ALIGNMENT);
    hm_states   = (uint64_t *) ALIGN_UP(h_UAstates, MEMORY_ALIGNMENT);
    checkCudaErrors(cudaHostRegister(hm_states, DATA_SIZE * sizeof(uint64_t), CU_MEMHOSTALLOC_DEVICEMAP));

    uint64_t* h_UAadvances = (uint64_t *) malloc (DATA_SIZE * sizeof(uint64_t) + MEMORY_ALIGNMENT);
    hm_advances   = (uint64_t *) ALIGN_UP(h_UAadvances, MEMORY_ALIGNMENT);
    checkCudaErrors(cudaHostRegister(hm_advances, DATA_SIZE * sizeof(uint64_t), CU_MEMHOSTALLOC_DEVICEMAP));

    unsigned int* h_UAfinished = (unsigned int *) malloc (DATA_SIZE * sizeof(unsigned int) + MEMORY_ALIGNMENT);
    hm_control   = (unsigned int *) ALIGN_UP(h_UAfinished, MEMORY_ALIGNMENT);
    checkCudaErrors(cudaHostRegister(hm_control, DATA_SIZE * sizeof(unsigned int), CU_MEMHOSTALLOC_DEVICEMAP));

    // map to device
    checkCudaErrors(cudaHostGetDevicePointer((void**)&d_advances, hm_advances, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void**)&d_control, hm_control, 0));
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
    cudaHostUnregister(hm_advances);
    cudaHostUnregister(hm_control);
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
        hm_control[i]         = CHAINSTATE_FINISHED;
        hm_states[i]          = 0;
        hm_advances[i]        = 0;
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
                currentJob->idle   = false;
                currentJob->cycles = 0;
                hm_advances[i]   = ReverseBits(currentJob->round_func[currentJob->start_round]);
                hm_states[i]       = ReverseBits(currentJob->start_value);
                hm_control[i]     = CHAINSTATE_RUNNING;
            } else {
            }
        } else {
            bool intMerge = currentJob->cycles > mMaxCycles;
            // int merge, if a chain went through the maximum cycles without reaching the finish state, just drop it
            if (intMerge) {
                hm_control[i]        = CHAINSTATE_FINISHED;
                currentJob->end_value = 0xffffffffffffffffULL;
                currentJob->idle      = true;
                controller->PushResult(currentJob);
                if (controller->PopRequest(currentJob)) { // <- pop request
                    hm_advances[i] = ReverseBits(currentJob->round_func[currentJob->start_round]);
                    hm_states[i]   = ReverseBits(currentJob->start_value);
                    hm_control[i] = CHAINSTATE_RUNNING;
                } else {
                }
            } else {
                if (hm_control[i] == CHAINSTATE_FINISHED) {
                    // reached the end round
                    if (currentJob->current_round == currentJob->end_round) {
                        // full search, rear submit
                        if (currentJob->end_round == mMaxRound - 1) {
                            currentJob->end_value = ReverseBits(hm_states[i]);
                            currentJob->idle      = true;
                            currentJob->cycles    = 0;
                            controller->PushResult(currentJob);
                            if (controller->PopRequest(currentJob)) {
                                hm_advances[i] = ReverseBits(currentJob->round_func[currentJob->start_round]);
                                hm_states[i]   = ReverseBits(currentJob->start_value);
                                hm_control[i] = CHAINSTATE_RUNNING;
                            }
                        // partial search, front submit
                        } else if (currentJob->end_round < mMaxRound - 1) {
                            currentJob->end_value = ReverseBits(hm_states[i]);
                            currentJob->idle      = true;
                            currentJob->cycles    = 0;
                            controller->PushResult(currentJob);
                            if (controller->PopRequest(currentJob)) {
                                hm_advances[i] = ReverseBits(currentJob->round_func[currentJob->start_round]);
                                hm_states[i]   = ReverseBits(currentJob->start_value);
                                hm_control[i] = CHAINSTATE_RUNNING;
                            }
                        }
                    // in process, just continue
                    } else if (currentJob->current_round < currentJob->end_round) {
                        currentJob->current_round   += 1;
                        currentJob->cycles            = 0;
                        hm_advances[i]                = ReverseBits(currentJob->round_func[currentJob->current_round]);
                        hm_control[i]                 = CHAINSTATE_RUNNING;
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
    // dec18_2014, huy.phung
    // dynamically specify block size


    dim3 dimBlock(BLOCK_SIZE, 1);
    dim3 dimGrid((DATA_SIZE - 1) / BLOCK_SIZE + 1, 1);
    a5_cuda_kernel<<<dimBlock, dimGrid>>>(mCycles, DATA_SIZE, mDp, d_states, d_advances, d_control);
    cudaDeviceSynchronize();
    // increase number of cycle for each slot
    for(unsigned int i = 0; i < DATA_SIZE; i++) {
        jobs[i].cycles += mCycles;
    }
    return;
}

/* Reverse bit order of an unsigned 64 bits int */
uint64_t A5CudaSlice::ReverseBits(uint64_t r)
{
  uint64_t r1 = r;
  uint64_t r2 = 0;
  for (int j = 0; j < 64 ; j++ ) {
    r2 = (r2<<1) | (r1 & 0x01);
    r1 = r1 >> 1;
  }
  return r2;
}


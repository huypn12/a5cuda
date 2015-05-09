#include <iostream>
#include <cstdio>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>

#include "utils/helper_cuda.h"
#include "A5CudaSlice.h"


void cuda_write_log(const char* str) {
    FILE* f_run;
    f_run = fopen("./cuda_log", "a");
    fprintf(f_run, "%s\n", str);
    fclose(f_run);
}

A5CudaSlice::A5CudaSlice(
        A5Cuda* a5cuda,
        int deviceId,
        int dp,
        unsigned int maxRound
        )
{
    // CUDA DEVICE RUNNING PARAMETERS
    mDeviceId = deviceId;
    mBlockSize = 128;
    mStreamCount = 8;
    mOffset = 512;
    mDataSize = mStreamCount * mOffset;
    mRunning = true;
    mController = a5cuda;
    mMaxRound  = maxRound;
    mIterate   = 128;
    mMaxCycles = 10*(2<<dp)*maxRound;
    mDp        = 32 - dp; // to deal with reversed bits 

    // Invoking running loop
    mRunningThread = new std::thread(&A5CudaSlice::workingLoop, this);
}

A5CudaSlice::~A5CudaSlice()
{
    halt();
    delete mRunningThread;
    cudaHostUnregister(hm_states);
    cudaHostUnregister(hm_control);
    free(mStreamArray);
    free(h_UAstates);
    free(h_UAcontrol);
    delete [] mJobs;
}

void A5CudaSlice::halt()
{
    mRunning = false;
    destroyStreams();
    mRunningThread->join();
}

void A5CudaSlice::workingLoop()
{
    checkCudaErrors( cudaSetDevice(mDeviceId) );
    checkCudaErrors( cudaGetDeviceProperties(&mCudaDeviceProp, mDeviceId) );
    init();
    while (mRunning) {
        probeStreams();
    }
    // Clean exit
    syncStreams();
}

void A5CudaSlice::init() {
    // Create streams to concurrently execute kernels
    mStreamArray = (cudaStream_t *) malloc(mStreamCount * sizeof(cudaStream_t));
    for (int i = 0; i < mStreamCount; i ++) {
        checkCudaErrors( cudaStreamCreate(&(mStreamArray[i])));
    }

    // Copying running parameters to constant memory
    unsigned int running_parameters[N_RUNNING_PARAM_CONSTANTS] = {
        mIterate,
        mDp,
        mOffset
    };
    checkCudaErrors(cudaMemcpyToSymbol(
                RUNNING_PARAM_CONSTANTS,
                (void*)running_parameters,
                4*sizeof(unsigned int)));

    // Memory allocation
    mJobs = new A5Cuda::JobPiece_s[mDataSize];

    uint4* h_UAstates = (uint4*) malloc(mDataSize*sizeof(uint4) + MEMORY_ALIGNMENT);
    hm_states = (uint4*) ALIGN_UP(h_UAstates, MEMORY_ALIGNMENT);
    checkCudaErrors(cudaHostRegister( hm_states, mDataSize*sizeof(uint4), CU_MEMHOSTALLOC_DEVICEMAP));
    checkCudaErrors(cudaHostGetDevicePointer( (void**)&d_states, hm_states, 0));

    unsigned int* h_UAcontrol = (unsigned int *) malloc(mDataSize * sizeof(unsigned int) + MEMORY_ALIGNMENT);
    hm_control   = (unsigned int *) ALIGN_UP(h_UAcontrol, MEMORY_ALIGNMENT);
    checkCudaErrors(cudaHostRegister( hm_control, mDataSize * sizeof(unsigned int), CU_MEMHOSTALLOC_DEVICEMAP));
    checkCudaErrors(cudaHostGetDevicePointer( (void**)&d_control, hm_control, 0));

    // Zeroing memory 
    for (unsigned int i = 0; i < mDataSize; i++)
    {
        mJobs[i].start_value   = 0;
        mJobs[i].end_value     = 0;
        mJobs[i].start_round   = 0;
        mJobs[i].current_round = 0;
        mJobs[i].end_round     = 0;
        mJobs[i].round_func    = NULL;
        mJobs[i].cycles        = 0;
        mJobs[i].idle          = true;
        hm_states[i].x         = 0;
        hm_states[i].y         = 0;
        hm_states[i].w         = 0;
        hm_states[i].z         = 0;
        hm_control[i]          = 0;//lc CHAINSTATE_control;
    }

}

void A5CudaSlice::probeStreams()
{
    for (int streamIdx=0; streamIdx < mStreamCount; streamIdx++) {
        if (cudaStreamQuery(mStreamArray[streamIdx]) == cudaSuccess) {
            process(streamIdx);
            invokeKernel(streamIdx);
        }
    }
}

void A5CudaSlice::syncStreams()
{
    for (int i = 0; i < mStreamCount; i++) {
        checkCudaErrors( cudaStreamSynchronize(mStreamArray[i]) );
    }
}

void A5CudaSlice::destroyStreams()
{
    for (int i = 0; i < mStreamCount; i++) {
        checkCudaErrors( cudaStreamDestroy(mStreamArray[i]) );
    }
}

void A5CudaSlice::process(int idx)
{
    int startIdx = idx * mOffset;
    int stopIdx = startIdx + mOffset;
    for (int i = startIdx; i < stopIdx; i++) {
        A5Cuda::JobPiece_s* currentJob = &mJobs[i];
        unsigned int control = 0;
        unsigned int dp_bits = (hm_states[i].y ^ hm_states[i].w) >> mDp;
        dp_bits = dp_bits | hm_control[i];
        uint64_t tempState = getValue(hm_states[i]);
        uint64_t tempRf    = getRf(hm_states[i]);
        if (!currentJob->idle) {
            currentJob->cycles += mIterate;
            bool intMerge = currentJob->cycles > mMaxCycles;
            if (intMerge) {
                currentJob->end_value = 0xffffffffffffffffULL;
                mController->PushResult(currentJob);
                currentJob->idle = true;
            } else {
                if(dp_bits == 0){
                    if (currentJob->current_round == currentJob->end_round) {
                        if (currentJob->end_round == mMaxRound - 1) {
                            currentJob->end_value = reversebits(tempState ^ tempRf, 64);
                            mController->PushResult(currentJob);
                            currentJob->idle = true;
                        } else if (currentJob->end_round < mMaxRound - 1) {
                            currentJob->end_value = reversebits(tempState ^ tempRf, 64);
                            mController->PushResult(currentJob);
                            currentJob->idle = true;
                        }
                    } else if (currentJob->current_round < currentJob->end_round) {
                        tempState                   = tempState ^ tempRf;
                        currentJob->current_round   += 1;
                        tempRf                      = currentJob->round_func[currentJob->current_round];
                        tempRf                      = reversebits(tempRf, 64);
                        tempState                   ^= tempRf;
                        currentJob->cycles = 0;
                        setValue(hm_states[i], tempState);
                        setRf(hm_states[i], tempRf);
                        control=1;
                    }
                }
            }
        }

        if (currentJob->idle) {
            if (mController->PopRequest(currentJob)) {
                currentJob->idle = false;
                currentJob->cycles = 0;
                setValueRev(hm_states[i], currentJob->start_value);
                setRfRev(hm_states[i], currentJob->round_func[currentJob->start_round]);
            }
        }
        hm_control[i]=control;
    }
}

// TODO: move mIterate to constant
void A5CudaSlice::invokeKernel(int idx)
{
    dim3 dimBlock(mBlockSize, 1);
    dim3 dimGrid((mOffset - 1) / mBlockSize + 1, 1);
    a51_cuda_kernel<<<dimBlock, dimGrid, 0, mStreamArray[idx]>>>(
            &(d_states[idx*mOffset]),//&(d_states[i*mOffset]),
            &(d_control[idx*mOffset])//&(d_control[i*mOffset])
            );
    return;
}

uint64_t A5CudaSlice::reversebits (uint64_t R, int length)
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

void A5CudaSlice::setValue(uint4 &slot, uint64_t value)
{
    slot.x = (unsigned int) value;
    slot.y = (unsigned int) (value >> 32);
}

void A5CudaSlice::setRf(uint4 &slot, uint64_t rf)
{
    slot.z = (unsigned int) rf;
    slot.w = (unsigned int) (rf >> 32);
}

uint64_t A5CudaSlice::getValue(uint4 state)
{
    return (((uint64_t) state.y << 32) | state.x);
}

uint64_t A5CudaSlice::getRf(uint4 state)
{
    return (((uint64_t) state.w << 32) | state.z);
}

void A5CudaSlice::setValueRev(uint4 &slot, uint64_t value)
{
    setValue(slot, reversebits(value, 64));
}

void A5CudaSlice::setRfRev(uint4 &slot, uint64_t rf)
{
    setRf(slot, reversebits(rf, 64));
}

uint64_t A5CudaSlice::getValueRev(uint4 state)
{
    return reversebits(((uint64_t) state.y << 32) | state.x, 64);
}

uint64_t A5CudaSlice::getRfRev(uint4 state)
{
    return reversebits(((uint64_t) state.w << 32) | state.z, 64);
}

/*void A5CudaSlice::tick()*/
/*{*/
    /*if (cudaStreamQuery(mCudaStream) == cudaSuccess) {*/
        /*process();*/
        /*invokeKernel();*/
    /*}*/
/*}*/



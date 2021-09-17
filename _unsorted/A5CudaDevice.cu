#include <iostream>
#include <cstdio>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>

#include "utils/helper_cuda.h"
#include "A5CudaDevice.h"


void cuda_write_log(const char* str) {
    FILE* f_run;
    f_run = fopen("./cuda_log", "a");
    fprintf(f_run, "%s\n", str);
    fclose(f_run);
}

A5CudaDevice::A5CudaDevice(
        A5Cuda* a5cuda,
        int deviceId,
        int dp,
        unsigned int maxRound
        )
{
    // Initialize CUDA device parameters
    // Each device has its own thread
    cudaSetDevice(deviceId);

    // Initialize running parameters
    mController = a5cuda;
    mIterate   = 128;
    mMaxRound  = maxRound;
    mMaxCycles = 10*(2<<dp)*maxRound;
    mDp        = 32 - dp; // to deal with reversed bits 
    mDataSize  = 512;
    mBlockSize = 128;
    mState     = eInit;
    mJobs      = new A5Cuda::JobPiece_s[mDataSize];

    uint4* h_UAstates = (uint4*) malloc(mDataSize*sizeof(uint4) + MEMORY_ALIGNMENT);
    hm_states = (uint4*) ALIGN_UP(h_UAstates, MEMORY_ALIGNMENT);
    checkCudaErrors(cudaHostRegister( hm_states, mDataSize*sizeof(uint4), CU_MEMHOSTALLOC_DEVICEMAP));
    checkCudaErrors(cudaHostGetDevicePointer( (void**)&d_states, hm_states, 0));

    unsigned int* h_UAcontrol = (unsigned int *) malloc(mDataSize * sizeof(unsigned int) + MEMORY_ALIGNMENT);
    hm_control   = (unsigned int *) ALIGN_UP(h_UAcontrol, MEMORY_ALIGNMENT);
    checkCudaErrors(cudaHostRegister( hm_control, mDataSize * sizeof(unsigned int), CU_MEMHOSTALLOC_DEVICEMAP));
    checkCudaErrors(cudaHostGetDevicePointer( (void**)&d_control, hm_control, 0));

    cudaStreamCreate(&mCudaStream);

    init();
}

A5CudaDevice::~A5CudaDevice()
{
    cudaStreamSynchronize(mCudaStream);
    cudaStreamDestroy(mCudaStream);
    cudaHostUnregister(hm_states);
    cudaHostUnregister(hm_control);
    delete [] mJobs;
}


void A5CudaDevice::tick()
{
    /*
    switch (mState)
    {
        case eInit:
            init();
            mState = eProcess;
            break;

        case eKernel:
            invokeKernel();
            mState = eProcess;
            break;

        case eProcess:
            process();
            mState = eKernel;
            break;

        default:
            std::cout << "state error" << std::endl;
            break;

    }
    */
    if (cudaStreamQuery(mCudaStream) == cudaSuccess) {
        process();
        invokeKernel();
    }
}


void A5CudaDevice::init()
{
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
        hm_control[i]          = 0;//lc CHAINSTATE_control;
        hm_states[i].x         = 0;
        hm_states[i].y         = 0;
        hm_states[i].w         = 0;
        hm_states[i].z         = 0;
    }
}

uint64_t A5CudaDevice::reversebits (uint64_t R, int length)
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

void A5CudaDevice::setValue(uint4 &slot, uint64_t value)
{
    slot.x = (unsigned int) value;
    slot.y = (unsigned int) (value >> 32);
}

void A5CudaDevice::setRf(uint4 &slot, uint64_t rf)
{
    slot.z = (unsigned int) rf;
    slot.w = (unsigned int) (rf >> 32);
}

uint64_t A5CudaDevice::getValue(uint4 state)
{
    return (((uint64_t) state.y << 32) | state.x);
}

uint64_t A5CudaDevice::getRf(uint4 state)
{
    return (((uint64_t) state.w << 32) | state.z);
}

void A5CudaDevice::setValueRev(uint4 &slot, uint64_t value)
{
    setValue(slot, reversebits(value, 64));
}

void A5CudaDevice::setRfRev(uint4 &slot, uint64_t rf)
{
    setRf(slot, reversebits(rf, 64));
}

uint64_t A5CudaDevice::getValueRev(uint4 state)
{
    return reversebits(((uint64_t) state.y << 32) | state.x, 64);
}

uint64_t A5CudaDevice::getRfRev(uint4 state)
{
    return reversebits(((uint64_t) state.w << 32) | state.z, 64);
}

void A5CudaDevice::process()
{

    for (unsigned int i = 0; i < mDataSize; i++) {
        A5Cuda::JobPiece_s* currentJob = &mJobs[i];
        currentJob->cycles += mIterate;

        unsigned int control = 0;
        unsigned int dp_bits = (hm_states[i].y ^ hm_states[i].w) >> mDp;
        dp_bits = dp_bits | hm_control[i];

        if (!currentJob->idle) {
            bool intMerge = currentJob->cycles > mMaxCycles;
            if (intMerge) {
                currentJob->end_value = 0xffffffffffffffffULL;
                mController->PushResult(currentJob);
                currentJob->idle = true;

            } else {
                if(dp_bits == 0){
                    uint64_t tempState = getValue(hm_states[i]);//((uint64_t) hm_states[i].y << 32) | hm_states[i].x;
                    uint64_t tempRf    = getRf(hm_states[i]);//((uint64_t) hm_states[i].w << 32) | hm_states[i].z;
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
                        /*tempState                   = reversebits(tempState, 64);*/
                        currentJob->current_round   += 1;
                        tempRf                      = currentJob->round_func[currentJob->current_round];
                        tempRf                      = reversebits(tempRf, 64);
                        tempState                   ^= tempRf;
                        /*setStateRev(hm_states[i], tempState, tempRf);*/
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
void A5CudaDevice::invokeKernel()
{
    dim3 dimBlock(mBlockSize, 1);
    dim3 dimGrid((mDataSize - 1) / mBlockSize + 1, 1);
    a51_cuda_kernel<<<dimBlock, dimGrid, 0, mCudaStream>>>(
            mIterate, mDataSize, mDp, d_states, d_control);

    //cudaDeviceSynchronize();
    return;
}


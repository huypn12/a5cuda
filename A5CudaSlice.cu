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
    // Initialize CUDA device parameters
    cudaSetDevice(deviceId);
#ifdef DEBUG
    printf("Running on device %d\n", deviceId);
#endif

    // Initialize running parameters
    mController = a5cuda;
    mIterate   = 128;
    mMaxRound  = maxRound;
    mMaxCycles = 10*(2<<dp)*maxRound;
    mDp        = 32 - dp; // to deal with reversed bits 
    mDataSize  = 4096;
    mBlockSize = 512;
    mState     = eInit;
    mJobs      = new A5Cuda::JobPiece_s[mDataSize];
#ifdef DEBUG
    printf(" mDP %d\n", mDp);
#endif

    uint4* h_UAstates = (uint4*) malloc(mDataSize*sizeof(uint4) + MEMORY_ALIGNMENT);
    hm_states = (uint4*) ALIGN_UP(h_UAstates, MEMORY_ALIGNMENT);
    checkCudaErrors(cudaHostRegister( hm_states, mDataSize*sizeof(uint4), CU_MEMHOSTALLOC_DEVICEMAP));
    checkCudaErrors(cudaHostGetDevicePointer( (void**)&d_states, hm_states, 0));

    unsigned int* h_UAcontrol = (unsigned int *) malloc(mDataSize * sizeof(unsigned int) + MEMORY_ALIGNMENT);
    hm_control   = (unsigned int *) ALIGN_UP(h_UAcontrol, MEMORY_ALIGNMENT);
    checkCudaErrors(cudaHostRegister( hm_control, mDataSize * sizeof(unsigned int), CU_MEMHOSTALLOC_DEVICEMAP));
    checkCudaErrors(cudaHostGetDevicePointer( (void**)&d_control, hm_control, 0));


}

A5CudaSlice::~A5CudaSlice()
{
    cudaHostUnregister(hm_states);
    cudaHostUnregister(hm_control);
    delete [] mJobs;
    cudaDeviceReset();
}


void A5CudaSlice::tick()
{
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
}


void A5CudaSlice::init()
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
        hm_control[i]         = 0;//lc CHAINSTATE_control;
        setStateZero(hm_states[i]);
    }
}

void A5CudaSlice::setState(
        uint4 &slot,
        uint64_t key,
        uint64_t advance
        )
{
    slot.x = (unsigned int) key;
    slot.y = (unsigned int) (key >> 32);
    slot.z = (unsigned int) advance;
    slot.w = (unsigned int) (advance >> 32);
}

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

void A5CudaSlice::setStateRev(
        uint4 &slot,
        uint64_t key,
        uint64_t advance
        )
{
    uint64_t revkey = reversebits(key, 64);
    uint64_t revadv = reversebits(advance, 64);

    slot.x = (unsigned int) revkey;
    slot.y = (unsigned int) (revkey >> 32);
    slot.z = (unsigned int) revadv;
    slot.w = (unsigned int) (revadv >> 32);
}

void A5CudaSlice::setValue(uint4 &slot, uint64_t value)
{
    slot.x = (unsigned int) value;
    slot.y = (unsigned int) (value >> 32);
}

void A5CudaSlice::setValueRev(uint4 &slot, uint64_t value)
{
    setValue(slot, reversebits(value, 64));
}

void A5CudaSlice::setRf(uint4 &slot, uint64_t rf)
{
    slot.x = (unsigned int) rf;
    slot.y = (unsigned int) (rf >> 32);
}
void A5CudaSlice::setRfRev(uint4 &slot, uint64_t rf)
{
    setValue(slot, reversebits(rf, 64));
}

uint64_t A5CudaSlice::getValue(uint4 state)
{
    return (((uint64_t) state.y << 32) | state.x);
}

uint64_t A5CudaSlice::getRf(uint4 state)
{
    return (((uint64_t) state.w << 32) | state.z);
}

uint64_t A5CudaSlice::getValueRev(uint4 state)
{
    return reversebits(((uint64_t) state.y << 32) | state.x, 64);
}

uint64_t A5CudaSlice::getRfRev(uint4 state)
{
    return reversebits(((uint64_t) state.w << 32) | state.z, 64);
}

void A5CudaSlice::setValueZero(uint4 &slot)
{
    slot.x = 0;
    slot.y = 0;
}

void A5CudaSlice::setRfZero(uint4 &slot)
{
    slot.z = 0;
    slot.w = 0;
}

void A5CudaSlice::setStateZero(
        uint4 &slot
        )
{
    slot.x = 0;
    slot.y = 0;
    slot.z = 0;
    slot.w = 0;
}

void A5CudaSlice::process()
{

    for (unsigned int i = 0; i < mDataSize; i++) {
        unsigned int control = 0;
        A5Cuda::JobPiece_s* currentJob = &mJobs[i];

        if (currentJob->idle) {
            if (mController->PopRequest(currentJob)) {
                currentJob->idle = false;
                setStateRev(
                        hm_states[i],
                        currentJob->start_value,
                        currentJob->round_func[currentJob->start_round]);
#ifdef DEBUG
                printf("start state=%x,%x  rf=%x,%x\n",
                        reversebits(hm_states[i].x, 32), reversebits(hm_states[i].y, 32),
                        reversebits(hm_states[i].z, 32), reversebits(hm_states[i].w, 32));
#endif
            } else {

            }
        } else {
            bool intMerge = currentJob->cycles > mMaxCycles;
            if (intMerge) {
                currentJob->end_value = 0xffffffffffffffffULL;
                mController->PushResult(currentJob);
                currentJob->idle = true;
                if (mController->PopRequest(currentJob)) {
                    currentJob->idle = false;
                    setStateRev(hm_states[i],
                            currentJob->start_value,
                            currentJob->round_func[currentJob->start_round]);
                } else {
                    setStateZero(hm_states[i]);
                }
            } else {
                unsigned int dp_bits = (hm_states[i].y ^ hm_states[i].w) >> mDp;
#ifdef DEBUG
#endif
                if(!dp_bits){
#ifdef DEBUG
                    printf("Chain finished\n");
                    printf("checking condition after start state=%x,%x  rf=%x,%x, %d\n",
                            reversebits(hm_states[i].x, 32), reversebits(hm_states[i].y, 32),
                            reversebits(hm_states[i].z, 32), reversebits(hm_states[i].w, 32),
                            hm_control[i]);
#endif
                    if (currentJob->current_round == currentJob->end_round) {
#ifdef DEBUG
                        printf("reach endround\n");
                        printf("checking condition after start state=%x,%x  rf=%x,%x, %d\n",
                                reversebits(hm_states[i].x, 32), reversebits(hm_states[i].y, 32),
                                reversebits(hm_states[i].z, 32), reversebits(hm_states[i].w, 32),
                                hm_control[i]);
#endif
                        if (currentJob->end_round == mMaxRound - 1) {
                            uint64_t tempState = ((uint64_t) hm_states[i].y << 32) | hm_states[i].x;
                            uint64_t tempRf    = ((uint64_t) hm_states[i].w << 32) | hm_states[i].z;
                            currentJob->end_value = reversebits(tempState ^ tempRf, 64);
#ifdef DEBUG
                            printf(" completed start state=%x,%x  rf=%x,%x\n",
                                    reversebits(hm_states[i].x, 32), reversebits(hm_states[i].y, 32),
                                    reversebits(hm_states[i].z, 32), reversebits(hm_states[i].w, 32));
#endif
                            mController->PushResult(currentJob);
                            currentJob->idle = true;
                            if (mController->PopRequest(currentJob)) {
                                currentJob->idle = false;
                                setStateRev(hm_states[i],
                                        currentJob->start_value,
                                        currentJob->round_func[currentJob->start_round]);
                            } else {
                                setStateZero(hm_states[i]);
                            }
                        } else if (currentJob->end_round < mMaxRound - 1) {
                            uint64_t tempState = ((uint64_t) hm_states[i].y << 32) | hm_states[i].x;
                            uint64_t tempRf    = ((uint64_t) hm_states[i].w << 32) | hm_states[i].z;
                            currentJob->end_value = reversebits(tempState ^ tempRf, 64);
#ifdef DEBUG
                            printf(" partial search start state=%x,%x  rf=%x,%x\n",
                                    reversebits(hm_states[i].x, 32), reversebits(hm_states[i].y, 32),
                                    reversebits(hm_states[i].z, 32), reversebits(hm_states[i].w, 32));
#endif
                            mController->PushResult(currentJob);
                            if (mController->PopRequest(currentJob)) {
                                currentJob->idle = false;
                                setStateRev(
                                        hm_states[i],
                                        currentJob->start_value,
                                        currentJob->round_func[currentJob->start_round]);
                            } else {
                                setStateZero(hm_states[i]);
                            }
                        }
                    } else if (currentJob->current_round < currentJob->end_round) {
#ifdef DEBUG
                        printf("switch round start state=%x,%x  rf=%x,%x\n",
                                reversebits(hm_states[i].x, 32), reversebits(hm_states[i].y, 32),
                                reversebits(hm_states[i].z, 32), reversebits(hm_states[i].w, 32));
#endif
                        uint64_t tempState          = ((uint64_t) hm_states[i].y) << 32 | hm_states[i].x;
                        uint64_t tempRf             = ((uint64_t) hm_states[i].w) << 32 | hm_states[i].z;
                        tempState                   = tempState ^ tempRf;
                        tempState                   = reversebits(tempState, 64);
                        if (!(tempState&0xFFF)) {
                        //    printf ("alarm\n");
                        }
#ifdef DEBUG
                        printf("switch round start %lx state=%x,%x  rf=%x,%x \n", tempState,
                                reversebits(hm_states[i].x, 32), reversebits(hm_states[i].y, 32),
                                reversebits(hm_states[i].z, 32), reversebits(hm_states[i].w, 32));
#endif
                        currentJob->current_round   += 1;
                        tempRf                      = currentJob->round_func[currentJob->current_round];
                        tempState                   ^= tempRf;
                        if (!(tempState&0xFFF)) {
                            printf ("alarm\n");
                        }
                        setStateRev(hm_states[i], tempState, tempRf);
#ifdef DEBUG
                        printf("switch round after start state=%x,%x  rf=%x,%x, dp=%lx %d\n",
                                reversebits(hm_states[i].x, 32), reversebits(hm_states[i].y, 32),
                                reversebits(hm_states[i].z, 32), reversebits(hm_states[i].w, 32),
                                (hm_states[i].y ^ hm_states[i].w) >> mDp,
                                hm_control[i]);
#endif
                        control=1;
                    }
                } else {
#ifdef DEBUG
                    printf("continue start state=%x,%x  rf=%x,%x, dp=%lx %d\n",
                            reversebits(hm_states[i].x, 32), reversebits(hm_states[i].y, 32),
                            reversebits(hm_states[i].z, 32), reversebits(hm_states[i].w, 32),
                            (hm_states[i].y ^ hm_states[i].w) >> mDp,
                            hm_control[i]);
#endif 
                }
            }
        }
        hm_control[i]=control;
    }
}

// TODO: move mIterate to constant
void A5CudaSlice::invokeKernel()
{
#ifdef DEBUG
    printf("kernel invoke here...\n");
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
#endif

    dim3 dimBlock(mBlockSize, 1);
    dim3 dimGrid((mDataSize - 1) / mBlockSize + 1, 1);
    a51_cuda_kernel<<<dimBlock, dimGrid>>>(mIterate, mDataSize, mDp, d_states, d_control);

#ifdef DEBUG
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time,start,stop);
    printf ("Time for the kernel: %f ms\n", time);
#else
    cudaDeviceSynchronize();
#endif
    for(unsigned int i = 0; i < mDataSize; i++) {
        mJobs[i].cycles += mIterate;
    }
    return;
}


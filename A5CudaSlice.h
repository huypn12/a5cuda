#include <stdint.h>

#include "A5Cuda.h"
#include "kernel/A5CudaKernel.h"

#define MEMORY_ALIGNMENT 4096
#define ALIGN_UP(x, size) (((size_t)x + (size - 1)) & (~(size - 1)))

class A5Cuda;

class A5CudaSlice {
    private:
        std::thread* mRunningThread;

        int mDeviceId;
        cudaDeviceProp mCudaDeviceProp;
        cudaStream_t *mStreamArray;
        unsigned int mStreamCount;
        unsigned int mDataSize;
        unsigned int mBlockSize;
        unsigned int mOffset;

        unsigned int mDp;
        unsigned int mMaxCycles;
        unsigned int mMaxRound;
        unsigned int mIterate;
        A5Cuda* mController;
        A5Cuda::JobPiece_s* mJobs;

        bool mRunning;

        uint4* h_UAstates;
        uint4* h_UAcontrol;
        uint4* hm_states;
        uint4* d_states;
        unsigned int* hm_control;
        unsigned int* d_control;

        uint64_t reversebits(uint64_t, int);
        void setValue(uint4 &slot, uint64_t value);
        void setRf(uint4 &slot, uint64_t rf);
        void setValueRev(uint4 &slot, uint64_t value);
        void setRfRev(uint4 &slot, uint64_t rf);
        uint64_t getValue(uint4 state);
        uint64_t getRf(uint4 state);
        uint64_t getValueRev(uint4 state);
        uint64_t getRfRev(uint4 state);

        void workingLoop();
        void init();
        void probeStreams();
        void syncStreams();
        void destroyStreams();
        void halt();
        void invokeKernel(int streamIdx);
        void process(int streamIdx);

    public:
        A5CudaSlice(A5Cuda* controller, int deviceId, int dp, unsigned int maxRound);
        ~A5CudaSlice();
        int getTotalSlots();
        int getAvailableSlots();
        int probeStream();
        void tick();
};

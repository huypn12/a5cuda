#include <stdint.h>

#include "A5Cuda.h"
#include "kernel/A5CudaKernel.h"

#define MEMORY_ALIGNMENT 4096
#define ALIGN_UP(x, size) (((size_t)x + (size - 1)) & (~(size - 1)))

class A5Cuda;

class A5CudaSlice {
    private:
        enum eState {
            eInit,
            eKernel,
            eProcess
        };

        eState mState;
        cudaStream_t mCudaStream;
        unsigned int mDp;
        unsigned int mMaxCycles;
        unsigned int mMaxRound;
        unsigned int mIterate;
        unsigned int mBlockSize;
        unsigned int mDataSize;
        A5Cuda* mController;
        A5Cuda::JobPiece_s* mJobs;

        bool isRunning;

        uint4* h_UAstates;
        uint4* h_UAcontrol;
        uint4* hm_states;
        uint4* d_states;
        unsigned int* hm_control;
        unsigned int* d_control;

        void setValue(uint4 &slot, uint64_t value);
        void setRf(uint4 &slot, uint64_t rf);
        void setValueRev(uint4 &slot, uint64_t value);
        void setRfRev(uint4 &slot, uint64_t rf);

        uint64_t getValue(uint4 state);
        uint64_t getRf(uint4 state);
        uint64_t getValueRev(uint4 state);
        uint64_t getRfRev(uint4 state);

        uint64_t reversebits(uint64_t, int);
        void init();
        void invokeKernel();
        void process();


    public:
        A5CudaSlice(A5Cuda* controller, int deviceId, int dp, unsigned int maxRound);
        ~A5CudaSlice();
        int getTotalSlots();
        int getAvailableSlots();
        int probeStream();
        void tick();
};

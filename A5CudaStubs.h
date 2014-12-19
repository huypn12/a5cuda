#ifndef A5_CUDA_STUBS
#define A5_CUDA_STUBS

#include <stdint.h>

bool A5CudaDeviceInfo(int& n_devices);
bool A5CudaInit(int max_rounds, int condition);
int A5CudaSubmit(uint64_t start_value, unsigned int start_round,
        uint32_t advance, void* context);
int A5CudaSubmitPartial(uint64_t start_value, unsigned int stop_round,
        uint32_t advance, void* context);
int A5CudaShutdown();
#endif

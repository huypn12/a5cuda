#include "A5CudaStubs.h"

#include <stdio.h>
#include <dlfcn.h>

static bool isDllLoaded = false;
static bool isDllError  = false;

static bool (*fPipelineInfo)(int &length) = NULL;
//static bool (*fInit)(int max, int cond, unsigned int mask, int mult) = NULL;
static bool (*fInit)(int max, int cond) = NULL;
static int  (*fSubmit)(uint64_t value, unsigned int start,
        uint32_t id, void* ctx) = NULL;
static int  (*fSubmitPartial)(uint64_t value, unsigned int stop,
        uint32_t id, void* ctx ) = NULL;
static bool (*fPopResult)(uint64_t& start, uint64_t& stop, void** ctx) = NULL;
static void (*fShutdown)(void) = NULL;

static bool LoadDllSym(void* handle, const char* name, void** func)
{
    *func = dlsym(handle, name);
    char* lError = dlerror();
    if (lError) {
        fprintf(stderr, "Error when loading symbol (%s): %s\n", name, lError);
        isDllError = true;
        return false;
    }
    return true;
}

static void LoadDLL(void)
{
    if (isDllError) return;

    void* lHandle = dlopen("./A5Cuda.so", RTLD_LAZY | RTLD_GLOBAL);

    char* lError = dlerror();
    if (lError) {
        fprintf(stderr, "Error when opening A5Cuda.so: %s\n", lError);
        return;
    }

//    LoadDllSym(lHandle, "A5CudaPipelineInfo", (void**)&fPipelineInfo);
    LoadDllSym(lHandle, "A5CudaInit", (void**)&fInit);
    LoadDllSym(lHandle, "A5CudaSubmit", (void**)&fSubmit);
    LoadDllSym(lHandle, "A5CudaSubmitPartial", (void**)&fSubmitPartial);
    LoadDllSym(lHandle, "A5CudaPopResult", (void**)&fPopResult);
    LoadDllSym(lHandle, "A5CudaShutdown", (void**)&fShutdown);

    isDllLoaded = !isDllError;
}


bool A5CudaInit(int max_rounds, int condition)//, unsigned int gpu_mask,int pipemul)
{
    LoadDLL();
    if (isDllLoaded) {
        return fInit(max_rounds, condition);//, gpu_mask, pipemul);
    } else {
        return false;
    }
}

bool A5CudaPipelineInfo(int &length)
{
    if (isDllLoaded) {
        return fPipelineInfo(length);
    } else {
        return false;
    }
}

int A5CudaSubmit(uint64_t start_value, unsigned int start_round,
        uint32_t advance, void* context)
{
    if (isDllLoaded) {
        return fSubmit(start_value, start_round, advance, context);
    } else {
        return -1;
    }
}

int  A5CudaSubmitPartial(uint64_t start_value, unsigned int stop_round,
        uint32_t advance, void* context)
{
    if (isDllLoaded) {
        return fSubmitPartial(start_value, stop_round, advance, context);
    } else {
        return -1;
    }
}

bool A5CudaPopResult(uint64_t& start_value, uint64_t& stop_value,
        void** context)
{
    if (isDllLoaded) {
        return fPopResult(start_value, stop_value, context);
    } else {
        return false;
    }
}

void A5CudaShutdown()
{
    if (isDllLoaded) {
        fShutdown();
    }
}


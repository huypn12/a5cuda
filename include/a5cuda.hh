#pragma once

#if defined _WIN32 || defined __CYGWIN__
#ifdef BUILDING_DLL
#ifdef __GNUC__
#define DLL_PUBLIC __attribute__((dllexport))
#else
#define DLL_PUBLIC __declspec(dllexport)
#endif
#else
#ifdef __GNUC__
#define DLL_PUBLIC __attribute__((dllimport))
#else
#define DLL_PUBLIC __declspec(dllimport)
#endif
#define DLL_LOCAL
#endif
#else
#if __GNUC__ >= 4
#define DLL_PUBLIC __attribute__((visibility("default")))
#define DLL_LOCAL __attribute__((visibility("hidden")))
#else
#define DLL_PUBLIC
#define DLL_LOCAL
#endif
#endif

#include "pch.hh"

bool A5Cuda_DeviceInfo(int n_devices);

bool A5Cuda_Init(int max_rounds, int condition);

int A5Cuda_Submit(uint64_t start_value, unsigned int start_round,
        uint32_t advance, void* context);

int A5Cuda_SubmitPartial(uint64_t start_value, unsigned int stop_round,
        uint32_t advance, void* context);

bool A5Cuda_PopResult(uint64_t& start_value, uint64_t& stop_value, void** dummy);

void A5Cuda_Shutdown();


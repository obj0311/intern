#pragma once
#include "MartianAIWrapper.h"
typedef unsigned long long int UINT64;

#define INTERFACE_API extern "C" __declspec(dllexport)

INTERFACE_API UINT64 Create();

INTERFACE_API int Destroy(UINT64 pInterface);

INTERFACE_API int Initialize(UINT64 pInterface, int deviceMode, int gpuSelection, WCHAR* modelPath);

INTERFACE_API int Run(UINT64 pInterface, unsigned char* inputImage, unsigned char* outputImage, InferenceInfo& inferenceInfo);


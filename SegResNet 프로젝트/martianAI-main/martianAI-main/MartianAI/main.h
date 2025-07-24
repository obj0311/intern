#pragma once
#define NOMINMAX

//#include "resource.h"
#include <windows.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "resource.h"
#include "ControlSharedMemory.h"
#include "InferenceControl.h"

//typedef struct
//{
//	int width;
//	int height;
//	int minRow;
//	int minColumn;
//	int maxRow;
//	int maxColumn;
//} ImageInfo;

struct InitSendInfo
{
	int m_InferenceDevics;
	int m_DeviceNum;
	int m_ModelPathLength;
	WCHAR m_ModelPath[1024];
};

const int MESSAGE_INITIALIZE = 1020;
const int MESSAGE_FINALIZE = 1022;
const int MESSAGE_RUN = 1024;
InferenceControl* g_InferenceControl = NULL;

template <typename Type>
inline void ClearBuffer(Type* buffer, size_t size)
{
	memset(buffer, 0x00, sizeof(Type) * size);
}

inline int Round(double input)
{
	return static_cast<int>(input + 0.5);
}
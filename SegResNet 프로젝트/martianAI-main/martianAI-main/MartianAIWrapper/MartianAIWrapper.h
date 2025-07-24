#pragma once
#include <Windows.h>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include "ControlSharedMemoryHost.h"

typedef enum
{
	enum_NO_ERROR = 0,
	enum_NULL_CHECK_ERROR,
	enum_DL_MODEL_FILE_ERROR,
	enum_CLASS_INIT_ERROR,
	enum_DL_INIT_ERROR,
	enum_DL_HANDLE_ERROR,
	enum_IPC_ERROR,
	enum_DATA_SIZE_ERROR,
	enum_DL_RUN_ERROR,
	enum_INPUT_SIZE_ERROR,
} ErrorValue;

class MartianAIWrapper
{
public:
	MartianAIWrapper(void);
	~MartianAIWrapper();

	int Run(unsigned char* InData, unsigned char* OutData, InferenceInfo& inferenceInfo);
	int Initialize(int deviceMode, int gpuSelection, WCHAR* modelPath);
	bool m_IsInitialized = false;

	struct InitSendInfo
	{
		int m_InferenceDevics;
		int m_DeviceNum;
		int m_ModelPathLength;
		WCHAR m_ModelPath[1024];
	};

private:
	std::string m_CurrentPath = "";
	const char* DLL_NAME = "MartianAIWrapper.dll";
	const char* EXE_PATH_NAME = "MartianAI\\";
	const char* EXE_NAME = "MartianAI.exe";
	const char* EXE_RECEIVER_NAME = "MARTIANAI";
	const char* EXE_WINDOW_NAME = "MartianAI";

	const int MESSAGE_INITIALIZE = 1020;
	const int MESSAGE_FINALIZE = 1022;
	const int MESSAGE_RUN = 1024;

	bool GetCoreHandle(void);
	int SendCopyDataInit(int mesesageNumber, InitSendInfo sendingData);
	int SendCopyDataRun(int mesesageNumber, void* sendingData, int dataSize, int dataCount);
	bool CloseCorePrograme(void);

	void GetCurrentBasePath(void);
	ControlSharedMemory* m_ControlSharedMemory = NULL;
};


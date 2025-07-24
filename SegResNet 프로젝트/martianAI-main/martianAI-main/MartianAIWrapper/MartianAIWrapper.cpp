#include "MartianAIWrapper.h"

MartianAIWrapper::MartianAIWrapper(void)
{
	GetCurrentBasePath();
	GetCoreHandle();
}

MartianAIWrapper::~MartianAIWrapper()
{
	CloseCorePrograme();
}

int MartianAIWrapper::Run(unsigned char* InData, unsigned char* OutData, InferenceInfo& inferenceInfo)
{
	// to-do Data Load and write

	int resultValue = 1;

	int sendingParameter[2];
	sendingParameter[0] = inferenceInfo.xSize * inferenceInfo.ySize * inferenceInfo.zSize;
	sendingParameter[1] = inferenceInfo.InferenceMode;

	if (m_ControlSharedMemory->WriteDataToSharedMemory(InData, inferenceInfo))
	{
		if (SendCopyDataRun(MESSAGE_RUN, sendingParameter, sizeof(int), 2) != 0)
		{
			resultValue = enum_DL_RUN_ERROR;
		}
		else
		{
			if (!m_ControlSharedMemory->ReadDataFromSharedMemory(OutData, inferenceInfo))
			{
				resultValue = enum_IPC_ERROR;
			}
			else
			{
				resultValue = enum_NO_ERROR;
			}
		}
	}

	return resultValue;
}


void MartianAIWrapper::GetCurrentBasePath(void)
{

	char szPath[1024];
	memset(szPath, 0x00, sizeof(szPath));
	int result = GetModuleFileNameA(GetModuleHandleA(DLL_NAME), szPath, 1024);
	std::string currentPath = szPath;
	m_CurrentPath.clear();
	const size_t last_slash_idx = currentPath.rfind('\\');
	if (std::string::npos != last_slash_idx)
	{
		m_CurrentPath = currentPath.substr(0, last_slash_idx);
	}
	m_CurrentPath.append("\\");
}

bool MartianAIWrapper::GetCoreHandle(void)
{
	bool resultValue = false;
	HWND windowHandle = NULL;
	if ((windowHandle = FindWindowA(EXE_RECEIVER_NAME, NULL)) == NULL && m_CurrentPath != "")
	{
		windowHandle = NULL;
		std::string applicationName(m_CurrentPath.c_str());
		//applicationName.append(EXE_PATH_NAME);
		applicationName.append(EXE_NAME);

		std::ifstream fileCheck(applicationName.c_str());
		if (fileCheck.good())
		{
			WinExec(applicationName.c_str(), SW_HIDE);
			//WinExec(applicationName.c_str(), SW_SHOWNORMAL);
			INT64 freq, from, to;
			QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
			QueryPerformanceCounter((LARGE_INTEGER*)&from);
			while ((windowHandle = FindWindowA(EXE_RECEIVER_NAME, NULL)) == NULL)
			{
				Sleep(10);
				QueryPerformanceCounter((LARGE_INTEGER*)&to);
				float elapseTime = static_cast<float>(to - from) / static_cast<float>(freq / 1000.0f);
				if (elapseTime > 300000) {
					break;
				}
			}
		}
	}

	return resultValue;
}

int MartianAIWrapper::SendCopyDataRun(int mesesageNumber, void* sendingData, int dataSize, int dataCount)
{
	int resultValue = -1;
	HWND windowHandle = FindWindowA(EXE_RECEIVER_NAME, NULL);
	if (windowHandle != NULL)
	{
		COPYDATASTRUCT sendData;
		sendData.dwData = 0;
		sendData.cbData = dataSize * dataCount;
		sendData.lpData = sendingData;

		resultValue = (int)SendMessageA(windowHandle, WM_COPYDATA, (WPARAM)mesesageNumber, (LPARAM)&sendData);
	}
	return resultValue;
}

int MartianAIWrapper::SendCopyDataInit(int mesesageNumber, InitSendInfo sendingData)
{
	int resultValue = -1;
	HWND windowHandle = FindWindowA(EXE_RECEIVER_NAME, NULL);
	if (windowHandle != NULL)
	{
		COPYDATASTRUCT sendData;
		sendData.dwData = 0;
		sendData.cbData = sizeof(sendingData);
		sendData.lpData = &sendingData;

		resultValue = (int)SendMessageA(windowHandle, WM_COPYDATA, (WPARAM)mesesageNumber, (LPARAM)&sendData);
	}
	return resultValue;
}

bool MartianAIWrapper::CloseCorePrograme(void)
{
	bool resultValue = 1;

	HWND windowHandle = FindWindowA(EXE_RECEIVER_NAME, NULL);
	if (windowHandle != NULL)
	{
		SendMessageA(windowHandle, WM_DESTROY, NULL, NULL);

		//Wait Untile the programe totally closed (limit time = 4s)
		INT64 freq, from, to;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		QueryPerformanceCounter((LARGE_INTEGER*)&from);
		while (FindWindowA(EXE_RECEIVER_NAME, NULL) != NULL)
		{
			Sleep(1);
			QueryPerformanceCounter((LARGE_INTEGER*)&to);
			float elapseTime = static_cast<float>(to - from) / static_cast<float>(freq / 1000.0f);
			if (elapseTime > 4000) break;
		}

		resultValue = 0;
		//if (m_ControlSharedMemory != NULL)
		//{
		//	delete m_ControlSharedMemory;
		//	m_ControlSharedMemory = NULL;
		//}
	}
	return resultValue;
}

int MartianAIWrapper::Initialize(int deviceMode, int gpuSelection, WCHAR* modelPath)
{
	int returnvalue = 0;

	InitSendInfo InitInfo = { 0, };

	InitInfo.m_InferenceDevics = deviceMode;
	InitInfo.m_DeviceNum = gpuSelection;
	InitInfo.m_ModelPathLength = wcslen(modelPath);
	//wmemset(InitInfo.m_ModelPath, 0, 1024);
	wmemcpy(InitInfo.m_ModelPath, modelPath, InitInfo.m_ModelPathLength);

	if (SendCopyDataInit(MESSAGE_INITIALIZE, InitInfo) == 0)
	{
		m_IsInitialized = true;
		if (m_ControlSharedMemory == NULL)
		{
			m_ControlSharedMemory = new ControlSharedMemory();
		}
		returnvalue = enum_NO_ERROR;
	}
	else
	{
		returnvalue = enum_DL_INIT_ERROR;
	}
	return returnvalue;
}
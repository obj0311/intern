#include "pch.h"
#include "MartianAIControl.h"

MartianAIControl::MartianAIControl()
{
	m_DllHandle = LoadLibraryA(DLL_NAME);
	if (m_DllHandle != NULL)
	{
		Interface_Create = (InterfaceType_Create)GetProcAddress(m_DllHandle, "Create");
		Interface_Destroy = (InterfaceType_Destroy)GetProcAddress(m_DllHandle, "Destroy");
		Interface_Initialize = (InterfaceType_Initialize)GetProcAddress(m_DllHandle, "Initialize");
		Interface_Run = (InterfaceType_Run)GetProcAddress(m_DllHandle, "Run");

		m_DllClassHandle = Interface_Create();
	}
}

MartianAIControl::~MartianAIControl()
{
	if (m_DllHandle != NULL)
	{
		Interface_Destroy(m_DllClassHandle);
		FreeLibrary(m_DllHandle);
	}
}

int MartianAIControl::Init(int deviceMode, int gpuSelection, WCHAR* modelPath)
{
	int resultValue = -1;

	if (m_DllHandle != NULL)
	{
		resultValue = Interface_Initialize(m_DllClassHandle, deviceMode, gpuSelection, modelPath);
	}
	return resultValue;
}


int MartianAIControl::Run(unsigned char* inputImage, unsigned char* outputImage, InferenceInfo& inferenceInfo)
{
	int resultValue = -1;

	if (m_DllHandle != NULL)
	{
		resultValue = Interface_Run(m_DllClassHandle, inputImage, outputImage, inferenceInfo);
	}

	return resultValue;
}

int MartianAIControl::CheckInitDone(int ModelNum)
{
	int resultValue = -1;
	WCHAR modelPath[10];
	if (m_DllHandle != NULL)
	{
		resultValue = Interface_Initialize(m_DllClassHandle, ModelNum, 0, modelPath);
	}
	return resultValue;
}
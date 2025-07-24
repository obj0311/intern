#include "WrapperInterface.h"

INTERFACE_API UINT64 Create()
{
	MartianAIWrapper* wrapper = new MartianAIWrapper();

	return reinterpret_cast<UINT64>(wrapper);
}

INTERFACE_API int Destroy(UINT64 pInterface)
{
	int resultValue = enum_DL_HANDLE_ERROR;
	if (pInterface > 0)
	{
		MartianAIWrapper* wrapper = reinterpret_cast<MartianAIWrapper*>(pInterface);
		delete wrapper;
		resultValue = enum_NO_ERROR;
	}

	return resultValue;
}

INTERFACE_API int Initialize(UINT64 pInterface, int deviceMode, int gpuSelection, WCHAR* modelPath)
{
	int resultValue = enum_DL_HANDLE_ERROR;
	if (pInterface > 0)
	{
		MartianAIWrapper* wrapper = reinterpret_cast<MartianAIWrapper*>(pInterface);

		resultValue = wrapper->Initialize(deviceMode, gpuSelection, modelPath);
	}

	return resultValue;
}


INTERFACE_API int Run(UINT64 pInterface, unsigned char* inputImage, unsigned char* outputImage, InferenceInfo& inferenceInfo)
{
	int resultValue = enum_DL_HANDLE_ERROR;
	if (pInterface > 0)
	{
		MartianAIWrapper* wrapper = reinterpret_cast<MartianAIWrapper*>(pInterface);

		resultValue = wrapper->Run(inputImage, outputImage, inferenceInfo);
	}

	return resultValue;
}

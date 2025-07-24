#pragma once
#include <Windows.h>
#define DLL_NAME "MartianAIWrapper.dll"

typedef struct
{
	int InferenceMode;
	int xSize;
	int ySize;
	int zSize;
	int processOnTable[16];
	float thresholdTable[16];
	int gaussianFilterMode;
	int kernelSize;
	int morphologyMode;
	int labelNum;
	int inferenceTime;
	int pcaTargetLabel;
	float pcaVector[9];
	float pcaMean[3];
	float SegMean[50];
	double imageRotate;
	int errorCode;
	int	returnValue;
} InferenceInfo;

class MartianAIControl
{
private:
	typedef UINT64(*InterfaceType_Create)();
	typedef int(*InterfaceType_Destroy)(UINT64);
	typedef int(*InterfaceType_Initialize)(UINT64, int, int, WCHAR*);
	typedef int(*InterfaceType_Run)(UINT64, unsigned char*, unsigned char*, InferenceInfo&);

	HMODULE m_DllHandle = NULL;
	INT64 m_DllClassHandle = -1;
	InterfaceType_Create Interface_Create = NULL;
	InterfaceType_Destroy Interface_Destroy = NULL;
	InterfaceType_Run Interface_Run = NULL;
	InterfaceType_Initialize Interface_Initialize = NULL;

public:
	MartianAIControl();
	~MartianAIControl();

	int Run(unsigned char* inputImage, unsigned char* outputImage, InferenceInfo& inferenceInfo);
	int Init(int deviceMode, int gpuSelection, WCHAR* modelPath);
	int CheckInitDone(int Model);

};
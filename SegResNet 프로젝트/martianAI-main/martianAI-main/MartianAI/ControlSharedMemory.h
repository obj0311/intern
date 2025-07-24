#pragma once
#define NOMINMAX

#include <Windows.h>
#include <string>

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

//typedef float DATA_TYPE;
typedef unsigned char DATA_TYPE;

const int MAX_DATA_SIZE = 256 * 256 * 256 * 2;
const std::string SHARED_MEMORY_NAME = "SharedName_MartianAI";

class ControlSharedMemory
{
private:
	unsigned char* m_SharedMemory = NULL;
	void* m_MapFile = NULL;

	//unsigned char* m_ImageInfoMemory = NULL;
	//unsigned char* m_ImageMemory = NULL;
	unsigned char* m_InferenceInfoMemory = NULL;
	unsigned char* m_DataMemory = NULL;

	int m_Width = 0;
	int m_Height = 0;

	size_t m_SizeSharedMemory = 0;
public:
	ControlSharedMemory();
	~ControlSharedMemory();

	bool ReadDataFromSharedMemory(DATA_TYPE* image, InferenceInfo& inferenceInfo);
	bool WriteDataToSharedMemory(DATA_TYPE* image, InferenceInfo& inferenceInfo);

private:
	template <typename Type>
	bool ReceiveThroughSharedMemory(Type* inputData, int count, unsigned char* address)
	{
		bool resultValue = false;
		if (inputData != NULL && address != NULL && count > 0)
		{
			try
			{
				size_t memorySize = sizeof(Type) * count;
				memcpy_s(inputData, memorySize, address, memorySize);
				resultValue = true;
			}
			catch (...)
			{
				resultValue = false;
			}
		}
		return resultValue;
	}

	template <typename Type>
	bool SendThroughSharedMemory(Type* inputData, int count, unsigned char* address)
	{
		bool resultValue = false;
		if (inputData != NULL && address != NULL && count > 0)
		{
			try
			{
				size_t memorySize = sizeof(Type) * count;
				memcpy_s(address, memorySize, inputData, memorySize);
				resultValue = true;
			}
			catch (...)
			{
				resultValue = false;
			}
		}
		return resultValue;
	}
};